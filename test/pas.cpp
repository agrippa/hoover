// Compile with -DHVR_BUCKET_SIZE=5 -DHVR_MAX_CONSTANT_ATTRS=10
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>

#include <shmem.h>
#include <hoover.h>

#define PATCH_GRAPH 0
#define AGENT_GRAPH 1
#define GRAPH_FEAT 9

#define PATCH_LAMBDA_H 0
#define PATCH_SV 1
#define PATCH_EV 2
#define PATCH_IV 3
#define PATCH_NV 4
#define PATCH_TIMESTEP 5
#define PATCH_NEXT_TIMESTEP_CREATED 6
#define PATCH_ID 7

#define AGENT_PATCH 0
#define AGENT_HOME_PATCH 1
#define AGENT_HEALTH 2
#define AGENT_ACTIVITY 3
#define AGENT_CURRENT_STAY 4
#define AGENT_TIMESTEP 5
#define AGENT_NEXT_TIMESTEP_CREATED 6
#define AGENT_ID 7

// Set to a tiny value so patch & agent can find each other iff they are at the same spot
const double connectivity_threshold = 0.01;

// Simulation space is square shaped, this value is not important
const double space_dim = 100.0;
// Put patches in square tiles, one patch per partition (they are used interchangeably in variable names)
const hvr_partition_t partition_dim = 64;
const double partition_side_len = space_dim / double(partition_dim);

// Agents are distributed evenly across all PEs
const uint64_t total_num_agents = 250000;
// Total number of initially infectious agents, also distributed evenly
const uint64_t total_num_init_Ih_agents = std::ceil(total_num_agents * 0.02);

// Four activities for each patch
// Do not change! Hard coded everywhere!
const uint64_t activities_per_patch = 4;
const double activities_maxstay[4] = {2, 4, 6, 8};
const double activities_alpha[4] = {1.0, 0.9, 0.8, 0.7};

// Number of days for the simulation
// const int n_days = 120;
const int n_days = 1;
// Number of agent time steps per day
// Changing this will affect dt_h/dt_v and therefore other parameters
const int agent_steps_per_day = 8;
// Number of patch time steps per agent time step
const int patch_steps_per_agent_step = 50;

const hvr_time_t max_agent_timestep = agent_steps_per_day * n_days;

const hvr_partition_t n_partitions = max_agent_timestep * partition_dim *
        partition_dim;

// ***** We only care about adult female mosquitoes *****
// Natural per capita emergence rate
const double psi_v = 0.3;
// Per capita death rate , 1/mu_v is the average life span
const double mu_v = 1.0 / 14.0;
// Intrinsic growth rate
const double r_v = psi_v - mu_v;
// Carrying capacity of a patch
const double Kv = 2500;
// # of times one mosquito would want to bite per unit time
const double sigma_v = 0.5;
// # of mosquito bites an average agent can sustain per unit time
const double sigma_h = 19;
// Probability of transmission of infection from an infectious mosquito to
// a susceptible agent
const double beta_hv = 0.33;
// Probability of transmission of infection from an infectious agent to a
// susceptible mosquito
const double beta_vh = 0.33;
// Per capita rate of progression of mosquitoes from the exposed state to
// the infectious state, 1/nu_v is the average duration of the latent period
const double nu_v = 0.1;
// Per capita rate of progression of agents from the exposed state to the
// infections state
const double nu_h = 1.0 / 5.0;
// Per capita recovery rate of agents
const double mu_h = 1.0 / 6.0;
// delta_t of the agent time steps
const double dt_h = 1.0 / agent_steps_per_day;
// delta_t of the patch time steps
const double dt_v = 1.0 / (agent_steps_per_day * patch_steps_per_agent_step);

// Probability of an exposed agent becomes infectious in a time step
const double PEtI = 1.0 - std::exp(-dt_h * nu_h);
// Probability of an infectious recovers in a time step
const double PItR = 1.0 - std::exp(-dt_h * mu_h);

// Run time constants that depends on mype and npes
int mype, npes;
uint64_t agents_per_pe, initial_infected_agents_per_pe;
hvr_partition_t patches_per_pe, patches_start, patches_end;

// High-quality RNG
std::random_device rd;
std::mt19937_64 rng(rd());

// RNG with a fixed seed, for free mode v.s. strict mode correctness checkings
// std::mt19937_64 rng(0);


// Adult female mosquito per capita emergence function (???)
// h_v(Nv,t) = (psi_v - r_v * Nv / Kv) * Nv
//           = [psi_v * (1 - Nv / Kv) + mu_v * Nv / Kv] * Nv
double h_v(const double psi_v, const double r_v, const double Nv, const double Kv) {
    return (psi_v - r_v * Nv / Kv) * Nv;
}


// Average force of infection to mosquitoes (rate of infection for each mosquito per unit time)
// Product of: the average number of bites per mosquito
//             the probability of transmission per bite
//             the probability that a bite is on an infectious agent
// ***** This is where the ABM and the ODE "communicates" *****
double lambda_v(const double sigma_v, const double sigma_h, const double beta_hv,
                const double Nv, const double Ihh, const double Nhh) {
    // Number of bites wanted by the mosquitoes
    double demand = sigma_v * Nv;
    // Number of bites could be sustained by the available agents
    double supply = sigma_h * Nhh;
    // # of successful bites
    double b = (demand * supply) / (demand + supply);

    return b / Nv * beta_hv * (Ihh / Nhh);
}


// Rate at which agents are infected from infectious mosquitoes
// Product of: the average number of bites a typical agent gets
//             the probability of transmission per bite
//             the probability that a bite is from an infectious mosquito
double lambda_h(const double sigma_v, const double sigma_h, const double beta_vh,
                const double Nv, const double Iv, const double Nhh) {
    // Number of bites wanted by the mosquitoes
    double demand = sigma_v * Nv;
    // Number of bites could be sustained by the available agents
    double supply = sigma_h * Nhh;
    // # of successful bites
    double b = (demand * supply) / (demand + supply);

    return b / Nhh * beta_vh * (Iv / Nv);
}


// Probability that a susceptible agent in a patch k becomes infected in a time step
double p_StE(const double lambda_h, const double dt_h) {
    return 1.0 - std::exp(-lambda_h * dt_h);
}


// Initialize global constants that depends on npes & mype
void init_constants() {
    mype = shmem_my_pe();
    npes = shmem_n_pes();
    agents_per_pe = std::ceil(double(total_num_agents) / double(npes));
    initial_infected_agents_per_pe = std::ceil(
            double(total_num_init_Ih_agents) / double(npes));
    patches_per_pe = ((partition_dim * partition_dim) + npes - 1) / npes;
    patches_start = patches_per_pe * mype;
    patches_end = patches_start + patches_per_pe;
    if (patches_end > partition_dim * partition_dim) {
        patches_end = partition_dim * partition_dim;
    }
}

static void patch_id_to_row_col(const hvr_partition_t patch, int &row,
        int &col) {
    row = patch / partition_dim;
    col = patch % partition_dim;
}

static hvr_partition_t row_col_to_patch_id(int row, int col) {
    return row * partition_dim + col;
}

// Initialize patch vertices, each patch has 15 attributes
void init_patches(hvr_ctx_t ctx) {
    // Constant attributes (10 of them):
    // 0: x coordinate  1: y coordinate
    // 7 ~ 10: maximum lengths of stay of activity 0 ~ 3
    // 11 ~ 14: relative risks of being bitten (alpha) of activity 0 ~ 3
    for (uint64_t i = 0; i < patches_per_pe; i++) {
        hvr_vertex_t *patch = hvr_vertex_create(ctx);

        int patch_id = patches_start + i;

        hvr_vertex_set(PATCH_ID, patch_id, patch, ctx);

        // Dynamic attributes (5 of them):
        // Store Ihh & Nhh for the record?
        hvr_vertex_set(PATCH_LAMBDA_H, 0.0, patch, ctx);       // lambda_h
        hvr_vertex_set(PATCH_SV, Kv / 2.0, patch, ctx);  // Sv
        hvr_vertex_set(PATCH_EV, 0.0, patch, ctx);       // Ev
        hvr_vertex_set(PATCH_IV, 0.0, patch, ctx);       // Iv
        hvr_vertex_set(PATCH_NV, Kv / 2.0, patch, ctx);  // Nv = Sv + Ev + Iv

        hvr_vertex_set(GRAPH_FEAT, PATCH_GRAPH, patch, ctx);

        hvr_vertex_set(PATCH_TIMESTEP, 0, patch, ctx);
        hvr_vertex_set(PATCH_NEXT_TIMESTEP_CREATED, 0, patch, ctx);
    }
}

// Initialize agent vertices, each agent has 7 attributes
void init_agents(hvr_ctx_t ctx) {
    // Constant attributes (10 of them):
    // 2: home x coordinate     3: home y coordinate

    // RNG to pick a random patch that belongs to this PE
    std::uniform_int_distribution<> pch_rng(patches_start, patches_end - 1);
    // RNG to pick a random activity between 0 ~ 3
    std::uniform_int_distribution<> act_rng(0, activities_per_patch - 1);

    for (uint64_t i = 0; i < agents_per_pe; i++) {
        hvr_vertex_t *agent = hvr_vertex_create(ctx);
        // Generate patch ID & activity ID
        const hvr_partition_t patch_id = pch_rng(rng);
        const uint64_t act_id = act_rng(rng);
        int agent_id = mype * agents_per_pe + i;

        hvr_vertex_set(AGENT_ID, agent_id, agent, ctx);
        hvr_vertex_set(AGENT_HOME_PATCH, patch_id, agent, ctx);
        hvr_vertex_set(AGENT_PATCH, patch_id, agent, ctx);

        // Health condition of an agent
        // 0.0:     susceptible
        // 1.0:     exposed
        // 2.0:     infectious
        // 3.0:     recovered (immune)
        double h = 0.0;

        // Make some of them infectious
        if (i < initial_infected_agents_per_pe) {
            h = 2.0;
        }

        // Dynamic attributes (5 of them):
        hvr_vertex_set(AGENT_HEALTH, h, agent, ctx);    // Health condition
        hvr_vertex_set(AGENT_ACTIVITY, double(act_id), agent, ctx);  // ID of current activity
        hvr_vertex_set(AGENT_CURRENT_STAY, 0.0, agent, ctx);  // Time steps stayed at current activity
        hvr_vertex_set(AGENT_TIMESTEP, 0, agent, ctx);
        hvr_vertex_set(AGENT_NEXT_TIMESTEP_CREATED, 0, agent, ctx);
        hvr_vertex_set(GRAPH_FEAT, AGENT_GRAPH, agent, ctx);
    }
}


/*
 * A (very clumsy) function to find the neighbors of a patch in the square tiles
 * of patches, given the row-major patch ID. The neighbors list include the
 * patch itself.
 */
void find_neighbor_patches(hvr_partition_t p_row, hvr_partition_t p_col,
        hvr_partition_t *out_rows, hvr_partition_t *out_cols,
        int &n_neighbors) {
    if (p_row == 0 && p_col == 0) {
        // Upper left corner
        n_neighbors = 4;
        out_rows[0] = p_row;     out_cols[0] = p_col;
        out_rows[1] = p_row;     out_cols[1] = p_col + 1;
        out_rows[2] = p_row + 1; out_cols[2] = p_col;
        out_rows[3] = p_row + 1; out_cols[3] = p_col + 1;
    } else if (p_row == partition_dim - 1 && p_col == partition_dim - 1) {
        // Lower right corner
        n_neighbors = 4;
        out_rows[0] = p_row;     out_cols[0] = p_col;
        out_rows[1] = p_row;     out_cols[1] = p_col - 1;
        out_rows[2] = p_row - 1; out_cols[2] = p_col;
        out_rows[3] = p_row - 1; out_cols[3] = p_col - 1;
    } else if (p_row == 0 && p_col == partition_dim - 1) {
        // Upper right corner
        n_neighbors = 4;
        out_rows[0] = p_row;     out_cols[0] = p_col;
        out_rows[1] = p_row;     out_cols[1] = p_col - 1;
        out_rows[2] = p_row + 1; out_cols[2] = p_col;
        out_rows[3] = p_row + 1; out_cols[3] = p_col - 1;
    } else if (p_row == partition_dim - 1 && p_col == 0) {
        // Lower left corner
        n_neighbors = 4;
        out_rows[0] = p_row;     out_cols[0] = p_col;
        out_rows[1] = p_row;     out_cols[1] = p_col + 1;
        out_rows[2] = p_row - 1; out_cols[2] = p_col;
        out_rows[3] = p_row - 1; out_cols[3] = p_col + 1;
    } else if (p_row == 0) {
        // Upper edge
        n_neighbors = 6;
        out_rows[0] = p_row;     out_cols[0] = p_col - 1;
        out_rows[1] = p_row;     out_cols[1] = p_col;
        out_rows[2] = p_row;     out_cols[2] = p_col + 1;
        out_rows[3] = p_row + 1; out_cols[3] = p_col - 1;
        out_rows[4] = p_row + 1; out_cols[4] = p_col;
        out_rows[5] = p_row + 1; out_cols[5] = p_col + 1;
    } else if (p_col == 0) {
        // Left edge
        n_neighbors = 6;
        out_rows[0] = p_row - 1; out_cols[0] = p_col;
        out_rows[1] = p_row - 1; out_cols[1] = p_col + 1;
        out_rows[2] = p_row;     out_cols[2] = p_col;
        out_rows[3] = p_row;     out_cols[3] = p_col + 1;
        out_rows[4] = p_row + 1; out_cols[4] = p_col;
        out_rows[5] = p_row + 1; out_cols[5] = p_col + 1;
    } else if (p_row == partition_dim - 1) {
        // Lower edge
        n_neighbors = 6;
        out_rows[0] = p_row;     out_cols[0] = p_col - 1;
        out_rows[1] = p_row;     out_cols[1] = p_col;
        out_rows[2] = p_row;     out_cols[2] = p_col + 1;
        out_rows[3] = p_row - 1; out_cols[3] = p_col - 1;
        out_rows[4] = p_row - 1; out_cols[4] = p_col;
        out_rows[5] = p_row - 1; out_cols[5] = p_col + 1;
    } else if (p_col == partition_dim - 1) {
        // Right edge
        n_neighbors = 6;
        out_rows[0] = p_row - 1; out_cols[0] = p_col;
        out_rows[1] = p_row - 1; out_cols[1] = p_col - 1;
        out_rows[2] = p_row;     out_cols[2] = p_col;
        out_rows[3] = p_row;     out_cols[3] = p_col - 1;
        out_rows[4] = p_row + 1; out_cols[4] = p_col;
        out_rows[5] = p_row + 1; out_cols[5] = p_col - 1;
    } else {                                            // Interior
        n_neighbors = 9;
        out_rows[0] = p_row - 1; out_cols[0] = p_col - 1;
        out_rows[1] = p_row - 1; out_cols[1] = p_col;
        out_rows[2] = p_row - 1; out_cols[2] = p_col + 1;
        out_rows[3] = p_row;     out_cols[3] = p_col - 1;
        out_rows[4] = p_row;     out_cols[4] = p_col;
        out_rows[5] = p_row;     out_cols[5] = p_col + 1;
        out_rows[6] = p_row + 1; out_cols[6] = p_col - 1;
        out_rows[7] = p_row + 1; out_cols[7] = p_col;
        out_rows[8] = p_row + 1; out_cols[8] = p_col + 1;
    }
}

/*
 * Given the home coordinates of an agent, generate a random destination that is
 * not too far away from home, return the new coordinates.
 */
void move_to_new_patch(const int home_patch, int& new_patch) {
    int home_row, home_col;
    patch_id_to_row_col(home_patch, home_row, home_col);

    // Find neighbors of the home patch
    int n_choices;
    hvr_partition_t choices_rows[9];
    hvr_partition_t choices_cols[9];
    find_neighbor_patches(home_row, home_col, choices_rows, choices_cols,
            n_choices);

    // Pick one from the neighbors, and return the coordinates
    std::uniform_int_distribution<> choice_rng(0, n_choices - 1);
    int choice_index = choice_rng(rng);
    int new_row = choices_rows[choice_index];
    int new_col = choices_cols[choice_index];
    new_patch = row_col_to_patch_id(new_row, new_col);
}

// Return the current patch/partition ID of an agent
hvr_partition_t actor_to_partition(const hvr_vertex_t *actor, hvr_ctx_t ctx) {
    int actor_type = (int)hvr_vertex_get(GRAPH_FEAT, actor, ctx);
    if (actor_type == PATCH_GRAPH) {
        return (hvr_partition_t)hvr_vertex_get(PATCH_ID, actor, ctx);
    } else {
        return (hvr_partition_t)hvr_vertex_get(AGENT_PATCH, actor, ctx);
    }
}

static void update_patch(hvr_vertex_t *patch, hvr_vertex_t **neighbors,
        hvr_edge_type_t *directions, int n_neighbors, hvr_ctx_t ctx) {
    const int timestep = hvr_vertex_get(PATCH_TIMESTEP, patch, ctx);

    if (timestep > 0) {
        hvr_vertex_t *prev = NULL;
        for (int i = 0; i < n_neighbors; i++) {
            hvr_vertex_t *neighbor = neighbors[i];
            int vertex_type = (int)hvr_vertex_get(GRAPH_FEAT, neighbor, ctx);
            if (vertex_type == PATCH_GRAPH) {
                int other_timestep = (int)hvr_vertex_get(PATCH_TIMESTEP,
                        neighbor, ctx);
                if (other_timestep == timestep - 1) {
                    assert(directions[i] == DIRECTED_IN);
                    assert(hvr_vertex_get(PATCH_ID, neighbor, ctx) ==
                            hvr_vertex_get(PATCH_ID, patch, ctx));
                    assert(prev == NULL);
                    prev = neighbor;
                }
            }
        }

        if (prev) {
            /*
             * This is a patch.
             * Calculate the total # of agents, and how many of them are
             * infectious for each activity in this patch.
             */
            uint64_t Nh[activities_per_patch] = {};
            uint64_t Ih[activities_per_patch] = {};

            for (size_t i = 0; i < n_neighbors; i++) {
                hvr_vertex_t *neighbor = neighbors[i];
                if ((int)hvr_vertex_get(GRAPH_FEAT, neighbor, ctx) !=
                        AGENT_GRAPH) {
                    continue;
                }

                assert(directions[i] == DIRECTED_IN);
                assert((int)hvr_vertex_get(AGENT_TIMESTEP, neighbor, ctx) ==
                        timestep - 1);

                const double h = hvr_vertex_get(AGENT_HEALTH, neighbor, ctx);
                const uint64_t act = hvr_vertex_get(AGENT_ACTIVITY, neighbor,
                        ctx);
                Nh[act] += 1;

                if (h < 2.5 && h > 1.5)     // Infectious!
                    Ih[act] += 1;
            }

            // Get mosquito population data from the last time step
            double Sv = hvr_vertex_get(PATCH_SV, prev, ctx);
            double Ev = hvr_vertex_get(PATCH_EV, prev, ctx);
            double Iv = hvr_vertex_get(PATCH_IV, prev, ctx);
            double Nv = hvr_vertex_get(PATCH_NV, prev, ctx);

            // Get relative risks of being bitten
            const double alpha_0 = activities_alpha[0];
            const double alpha_1 = activities_alpha[1];
            const double alpha_2 = activities_alpha[2];
            const double alpha_3 = activities_alpha[3];

            // Calculate effective numbers of agents
            const double Nhh = Nh[0] * alpha_0 + Nh[1] * alpha_1 +
                Nh[2] * alpha_2 + Nh[3] * alpha_3;
            const double Ihh = Ih[0] * alpha_0 + Ih[1] * alpha_1 +
                Ih[2] * alpha_2 + Ih[3] * alpha_3;

            // Compute the mosquitoes' life cycles: ODE solving with RK4
            // Maybe I should compute lambda_v & h_v only once?
            for (int ts_v = 0; ts_v < patch_steps_per_agent_step; ts_v++) {
                double Sv_k1 = dt_v * (h_v(psi_v, r_v, Nv, Kv) - lambda_v(sigma_v,
                            sigma_h, beta_hv, Nv, Ihh, Nhh) * Sv - mu_v * Sv);
                double Sv_k2 = dt_v * (h_v(psi_v, r_v, Nv, Kv) -
                        lambda_v(sigma_v, sigma_h, beta_hv, Nv, Ihh, Nhh) *
                        (Sv + Sv_k1/2) - mu_v * (Sv + Sv_k1/2));
                double Sv_k3 = dt_v * (h_v(psi_v, r_v, Nv, Kv) -
                        lambda_v(sigma_v, sigma_h, beta_hv, Nv, Ihh, Nhh) *
                        (Sv + Sv_k2/2) - mu_v * (Sv + Sv_k2/2));
                double Sv_k4 = dt_v * (h_v(psi_v, r_v, Nv, Kv) -
                        lambda_v(sigma_v, sigma_h, beta_hv, Nv, Ihh, Nhh) *
                        (Sv + Sv_k3) - mu_v * (Sv + Sv_k3));
                Sv += (Sv_k1 + 2 * (Sv_k2 + Sv_k3) + Sv_k4) / 6;

                double Ev_k1 = dt_v * (lambda_v(sigma_v, sigma_h, beta_hv, Nv, Ihh,
                            Nhh) * Sv - nu_v * Ev - mu_v * Ev);
                double Ev_k2 = dt_v *
                    (lambda_v(sigma_v, sigma_h, beta_hv, Nv, Ihh, Nhh) *
                     (Sv + Sv_k1/2) - nu_v * (Ev + Ev_k1/2) - mu_v *
                     (Ev + Ev_k1/2));
                double Ev_k3 = dt_v *
                    (lambda_v(sigma_v, sigma_h, beta_hv, Nv, Ihh, Nhh) *
                     (Sv + Sv_k2/2) - nu_v * (Ev + Ev_k2/2) - mu_v *
                     (Ev + Ev_k2/2));
                double Ev_k4 = dt_v *
                    (lambda_v(sigma_v, sigma_h, beta_hv, Nv, Ihh, Nhh) *
                     (Sv + Sv_k3) - nu_v * (Ev + Ev_k3) - mu_v * (Ev + Ev_k3));
                Ev += (Ev_k1 + 2 * (Ev_k2 + Ev_k3) + Ev_k4) / 6;

                double Iv_k1 = dt_v * (nu_v * Ev - mu_v * Iv);
                double Iv_k2 = dt_v * (nu_v * (Ev + Ev_k1 / 2) - mu_v *
                        (Iv + Iv_k1 / 2));
                double Iv_k3 = dt_v * (nu_v * (Ev + Ev_k2 / 2) - mu_v *
                        (Iv + Iv_k2 / 2));
                double Iv_k4 = dt_v * (nu_v * (Ev + Ev_k3) - mu_v * (Iv + Iv_k3));
                Iv += (Iv_k1 + 2 * (Iv_k2 + Iv_k3) + Iv_k4) / 6;

                // Total number of mosquitoes
                Nv = Sv + Ev + Iv;
            }

            // Compute mosquito-to-host force of infection lambda_h
            const double lh = lambda_h(sigma_v, sigma_h, beta_vh, Nv, Iv, Nhh);

            hvr_vertex_set(PATCH_LAMBDA_H, lh, patch, ctx);
            hvr_vertex_set(PATCH_SV, Sv, patch, ctx);
            hvr_vertex_set(PATCH_EV, Ev, patch, ctx);
            hvr_vertex_set(PATCH_IV, Iv, patch, ctx);
            hvr_vertex_set(PATCH_NV, Nv, patch, ctx);
        }
    }

    if ((int)hvr_vertex_get(PATCH_TIMESTEP, patch, ctx) <
                max_agent_timestep - 1 &&
            (int)hvr_vertex_get(PATCH_NEXT_TIMESTEP_CREATED, patch, ctx) == 0) {
        hvr_vertex_t *next = hvr_vertex_create(ctx);
        assert(next);

        hvr_vertex_set(GRAPH_FEAT, PATCH_GRAPH, next, ctx);
        hvr_vertex_set(PATCH_ID,                    hvr_vertex_get(PATCH_ID, patch, ctx),             next, ctx);
        hvr_vertex_set(PATCH_LAMBDA_H,              hvr_vertex_get(PATCH_LAMBDA_H, patch, ctx),       next, ctx);
        hvr_vertex_set(PATCH_SV,                    hvr_vertex_get(PATCH_SV, patch, ctx),             next, ctx);
        hvr_vertex_set(PATCH_EV,                    hvr_vertex_get(PATCH_EV, patch, ctx),             next, ctx);
        hvr_vertex_set(PATCH_IV,                    hvr_vertex_get(PATCH_IV, patch, ctx),             next, ctx);
        hvr_vertex_set(PATCH_NV,                    hvr_vertex_get(PATCH_NV, patch, ctx),             next, ctx);
        hvr_vertex_set(PATCH_TIMESTEP,              1 + hvr_vertex_get(PATCH_TIMESTEP, patch, ctx), next, ctx);
        hvr_vertex_set(PATCH_NEXT_TIMESTEP_CREATED, 0, next, ctx);

        hvr_vertex_set(PATCH_NEXT_TIMESTEP_CREATED, 1, patch, ctx);
    }
}

static void update_agent(hvr_vertex_t *agent, hvr_vertex_t **neighbors,
        hvr_edge_type_t *directions, int n_neighbors, hvr_ctx_t ctx) {
    const int timestep = hvr_vertex_get(AGENT_TIMESTEP, agent, ctx);
    const int agent_id = hvr_vertex_get(AGENT_ID, agent, ctx);

    if (timestep > 0) {
        // Find my patch and my previou state
        hvr_vertex_t* patch = nullptr;
        hvr_vertex_t *prev = NULL;

        for (size_t i = 0; i < n_neighbors; i++) {
            hvr_vertex_t *neighbor = neighbors[i];
            if ((int)hvr_vertex_get(GRAPH_FEAT, neighbor, ctx) == PATCH_GRAPH) {
                assert(patch == NULL);
                // Think this might not be true?
                assert(directions[i] == DIRECTED_IN);
                assert((int)hvr_vertex_get(PATCH_TIMESTEP, neighbor, ctx) ==
                        timestep - 1);
                patch = neighbor;
            } else {
                assert((int)hvr_vertex_get(GRAPH_FEAT, neighbor, ctx) ==
                        AGENT_GRAPH);
                int other_timestep = (int)hvr_vertex_get(AGENT_TIMESTEP,
                        neighbor, ctx);
                int other_agent_id = (int)hvr_vertex_get(AGENT_ID, neighbor,
                        ctx);

                if (other_agent_id == agent_id &&
                        other_timestep == timestep - 1) {
                    assert(directions[i] == DIRECTED_IN);
                    assert(prev == NULL);
                    prev = neighbor;
                }
                
            }
        }

        if (!patch || !prev) {
            return;
        }

        // Update health status
        const double health = hvr_vertex_get(AGENT_HEALTH, prev, ctx);
        const int act_id = hvr_vertex_get(AGENT_ACTIVITY, prev, ctx);
        const double alpha = activities_alpha[act_id];

        // Generate a random number for health status transition
        std::uniform_real_distribution<> h_trans(0.0, 1.0);
        const double p = h_trans(rng);

        if (health < 0.5) {         // susceptible
            // Get lambda_h
            const double lh = hvr_vertex_get(PATCH_LAMBDA_H, patch, ctx);
            // Calculate the probability of susceptible-to-exposed
            const double PStE = p_StE(alpha * lh, dt_h);
            if (p < PStE)           // Infect this agent
                hvr_vertex_set(AGENT_HEALTH, 1.0, agent, ctx);
        } else if (health < 1.5) {  // Exposed
            if (p < PEtI)           // Become infectious
                hvr_vertex_set(AGENT_HEALTH, 2.0, agent, ctx);
        } else if (health < 2.5) {  // Infectious
            if (p < PItR)           // Recover from the disease
                hvr_vertex_set(AGENT_HEALTH, 3.0, agent, ctx);
        } else {
            // Immune
        }

        // Now decide if the agent should move to a new location
        const double stayed = hvr_vertex_get(AGENT_CURRENT_STAY, prev, ctx);
        const double max_stay = activities_maxstay[act_id];

        if (stayed >= max_stay - 0.001) {
            const int home_patch = (int)hvr_vertex_get(AGENT_HOME_PATCH, prev,
                    ctx);
            int new_patch;
            move_to_new_patch(home_patch, new_patch);
            hvr_vertex_set(AGENT_PATCH, new_patch, agent, ctx);

            // Moved to a new patch, now generate a random activity ID
            std::uniform_int_distribution<> act_rng(0, activities_per_patch - 1);
            hvr_vertex_set(AGENT_HEALTH, double(act_rng(rng)), agent, ctx);

            // Clear the # of time steps stayed at current activity
            hvr_vertex_set(AGENT_CURRENT_STAY, 0.0, agent, ctx);
        } else {
            // Increment the # of time steps stayed at current activity
            hvr_vertex_set(AGENT_CURRENT_STAY, stayed + 1.0, agent, ctx);
        }
    }

    if ((int)hvr_vertex_get(AGENT_TIMESTEP, agent, ctx) <
                max_agent_timestep - 1 &&
            (int)hvr_vertex_get(AGENT_NEXT_TIMESTEP_CREATED, agent, ctx) == 0) {
        hvr_vertex_t *next = hvr_vertex_create(ctx);
        assert(next);

        hvr_vertex_set(GRAPH_FEAT, AGENT_GRAPH, next, ctx);
        hvr_vertex_set(AGENT_ID, hvr_vertex_get(AGENT_ID, agent, ctx), next, ctx);
        hvr_vertex_set(AGENT_PATCH, hvr_vertex_get(AGENT_PATCH, agent, ctx), next, ctx);
        hvr_vertex_set(AGENT_HOME_PATCH, hvr_vertex_get(AGENT_HOME_PATCH, agent, ctx), next, ctx);
        hvr_vertex_set(AGENT_HEALTH, hvr_vertex_get(AGENT_HEALTH, agent, ctx), next, ctx);
        hvr_vertex_set(AGENT_ACTIVITY, hvr_vertex_get(AGENT_ACTIVITY, agent, ctx), next, ctx);
        hvr_vertex_set(AGENT_CURRENT_STAY, hvr_vertex_get(AGENT_CURRENT_STAY, agent, ctx), next, ctx);
        hvr_vertex_set(AGENT_TIMESTEP, 1 + hvr_vertex_get(AGENT_TIMESTEP, agent, ctx), next, ctx);
        hvr_vertex_set(AGENT_NEXT_TIMESTEP_CREATED, 0, next, ctx);

        hvr_vertex_set(AGENT_NEXT_TIMESTEP_CREATED, 1, agent, ctx);
    }
}

void update_metadata(hvr_vertex_t *vec,
                     hvr_set_t *couple_with,
                     hvr_ctx_t ctx) {
    // Only useful for printing debug info about # of neighbors
    hvr_vertex_t **neighbors;
    hvr_edge_type_t *directions;
    int n_neighbors = hvr_get_neighbors(vec, &neighbors, &directions, ctx);

    if ((int)hvr_vertex_get(GRAPH_FEAT, vec, ctx) == PATCH_GRAPH) {
        update_patch(vec, neighbors, directions, n_neighbors, ctx);
    } else {
        assert((int)hvr_vertex_get(GRAPH_FEAT, vec, ctx) == AGENT_GRAPH);
        update_agent(vec, neighbors, directions, n_neighbors, ctx);
    }
}

// We only interact with neighboring patches/partitions
void might_interact(const hvr_partition_t partition,
                   hvr_partition_t *interacting_partitions,
                   unsigned *n_interacting_partitions,
                   const unsigned interacting_partitions_capacity,
                   hvr_ctx_t ctx) {
    /*
     * Given a partition (time, row, col) it can interact with
     * (time - 1, row, col) or (time + 1, row, col)
     */
    hvr_partition_t t = partition / (partition_dim * partition_dim);
    hvr_partition_t row = (partition / partition_dim) % partition_dim;
    hvr_partition_t col = partition % partition_dim;

    int n_choices;
    hvr_partition_t choices_rows[9];
    hvr_partition_t choices_cols[9];
    find_neighbor_patches(row, col, choices_rows, choices_cols, n_choices);

    int n = 0;
    if (t > 0) {
        for (int c = 0; c < n_choices; c++) {
            interacting_partitions[n++] =
                (t - 1) * partition_dim * partition_dim +
                choices_rows[c] * partition_dim + choices_cols[c];
        }
    }

    if (t < max_agent_timestep - 1) {
        for (int c = 0; c < n_choices; c++) {
            interacting_partitions[n++] =
                (t + 1) * partition_dim * partition_dim +
                choices_rows[c] * partition_dim + choices_cols[c];
        }
    }

    *n_interacting_partitions = n;
}

// We do some statistics here, and we do not abort
void update_coupled_val(hvr_vertex_iter_t *iter,
                hvr_ctx_t ctx,
                hvr_vertex_t *out_coupled_metric) {
    // Dummy value, unused
    hvr_vertex_set(0, 0.0, out_coupled_metric, ctx);

#if 0
    uint64_t Sh = 0.0;
    uint64_t Eh = 0.0;
    uint64_t Ih = 0.0;
    uint64_t Rh = 0.0;

    // Count the number of each type of the agents
    for (hvr_vertex_t *v = hvr_vertex_iter_next(iter); v;
            v = hvr_vertex_iter_next(iter)) {
        if ((int)hvr_vertex_get(GRAPH_FEAT, v, ctx) == AGENT_GRAPH) {
            const double health = hvr_vertex_get(4, v, ctx);
            if (health < 0.5) {
                Sh += 1;
            } else if (health < 1.5) {
                Eh += 1;
            } else if (health < 2.5) {
                Ih += 1;
            } else {
                Rh += 1;
            }
        }
    }

    // Print agent statistics in CSV format
    std::cout << ctx->iter << ", " << ctx->pe
              << ", " << Sh << ", " << Eh
              << ", " << Ih << ", " << Rh << '\n';
#endif
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric, // coupled_pes[shmem_my_pe()]
        hvr_vertex_t *all_coupled_metrics, // Each PE's val
        hvr_vertex_t *global_coupled_metric, // Sum reduction of coupled_pes
        hvr_set_t *coupled_pes, // An array of size npes, with each PE's val
        int n_coupled_pes,
        int *updates_on_this_iter, // An array of size npes, the number of vertex updates done on each coupled PE
        hvr_set_t *terminated_coupled_pes) {
    int sum_updates = 0;
    for (int i = 0; i < ctx->npes; i++) {
        sum_updates += updates_on_this_iter[i];
    }
    printf("PE %d saw %d updates from %d coupled PEs on iter %d\n",
            mype, sum_updates, n_coupled_pes, ctx->iter);

    return 0;
}

/*
 * Each agent and patch should have an edge with its previous and next state.
 *
 * Each patch should also have edges with all vertices in its patch on the prior
 * time step.
 *
 * Each agent should have edges with their current patch on its next timestep.
 */
hvr_edge_type_t should_have_edge(const hvr_vertex_t *base,
        const hvr_vertex_t *neighbor,
        hvr_ctx_t ctx) {
    int base_type =     (int)hvr_vertex_get(GRAPH_FEAT, base, ctx);
    int neighbor_type = (int)hvr_vertex_get(GRAPH_FEAT, neighbor, ctx);

    int base_timestep, neighbor_timestep;
    if (base_type == PATCH_GRAPH) {
        base_timestep = hvr_vertex_get(PATCH_TIMESTEP, base, ctx);
        neighbor_timestep = hvr_vertex_get(PATCH_TIMESTEP, neighbor, ctx);
    } else {
        assert(base_type == AGENT_GRAPH);
        base_timestep = hvr_vertex_get(AGENT_TIMESTEP, base, ctx);
        neighbor_timestep = hvr_vertex_get(AGENT_TIMESTEP, neighbor, ctx);
    }

    if (base_type == neighbor_type) {
        int base_id, neighbor_id;
        if (base_type == PATCH_GRAPH) {
            base_id = hvr_vertex_get(PATCH_ID, base, ctx);
            neighbor_id = hvr_vertex_get(PATCH_ID, neighbor, ctx);
        } else {
            base_id = hvr_vertex_get(AGENT_ID, base, ctx);
            neighbor_id = hvr_vertex_get(AGENT_ID, neighbor, ctx);
        }

        if (base_id == neighbor_id &&
                abs(neighbor_timestep - base_timestep) == 1) {
            if (base_timestep < neighbor_timestep) {
                return DIRECTED_OUT;
            } else {
                return DIRECTED_IN;
            }
        }
    } else {
        const hvr_vertex_t *agent = (base_type == AGENT_GRAPH ? base : neighbor);
        const hvr_vertex_t *patch = (base_type == PATCH_GRAPH ? base : neighbor);
        int agent_timestep = (int)hvr_vertex_get(AGENT_TIMESTEP, agent, ctx);
        int patch_timestep = (int)hvr_vertex_get(AGENT_TIMESTEP, patch, ctx);

        hvr_partition_t patch_id = (hvr_partition_t)hvr_vertex_get(AGENT_PATCH,
                agent, ctx);

        if (patch_id == (int)hvr_vertex_get(PATCH_ID, patch, ctx) &&
                abs(neighbor_timestep - base_timestep) == 1) {
            if (base_timestep < neighbor_timestep) {
                return DIRECTED_OUT;
            } else {
                return DIRECTED_IN;
            }
        }
    }

    return NO_EDGE;
}

int main() {
    shmem_init();

    // Initialize global constants
    init_constants();

    hvr_ctx_t ctx;
    hvr_ctx_create(&ctx);

    init_patches(ctx);

    init_agents(ctx);

    hvr_init(n_partitions,
            update_metadata,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            NULL,
            should_have_edge,
            should_terminate,
            20,
            1,
            ctx);

    shmem_barrier_all();

    auto t_start = std::chrono::steady_clock::now();

    hvr_body(ctx);

    shmem_barrier_all();

    auto t_end = std::chrono::steady_clock::now();

    hvr_finalize(ctx);

    shmem_finalize();

    // Print timing info
    if (mype == 0)
        std::cout << "Time elapsed: "
                  << std::chrono::duration<double>(t_end - t_start).count()
                  << " seconds\n";
}
