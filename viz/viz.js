var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
var simulation_data = {}

console.log('Canvas = ' + canvas.width + ' x ' + canvas.height);

var min_x = null;
var min_y = null;
var max_x = null;
var max_y = null;

var curr_simulation_step = 1000000;
var starting_simulation_step;

var color_based_on_state = true;
var pe_colors = {}
var state_colors = {}

var random_colors_index = 0;
var random_colors = ['rgb(51,255,51)', /* 'rgb(0,0,255)', */ 'rgb(255,0,0)',
    'rgb(255,255,0)', 'rgb(204,0,204)', 'rgb(0,255,255)'];

//Helper function to get a random color - but not too dark
function GetRandomColor() {
    var result = random_colors[random_colors_index];
    random_colors_index += 1;
    if (random_colors_index >= random_colors.length) {
        random_colors_index = 0;
    }
    return result;
}

function addPE(pe) {
    if (pe in pe_colors) {
        return;
    }
    pe_colors[pe] = GetRandomColor();
}

function addState(state) {
    if (state in state_colors) {
        return
    }
    state_colors[state] = GetRandomColor();
}

function getStateColor(state) {
    return state_colors[state];
}

function getPEColor(pe) {
    return pe_colors[pe];
}

//Particle object with random starting position, velocity and color
var Particle = function (set_x, set_y, set_pe, set_state) {
    this.x = set_x;
    this.y = set_y;
    this.pe = set_pe;
    this.state = set_state;
}

function changeColoring(e) {
    if (e.checked) {
        color_based_on_state = true;
    } else {
        color_based_on_state = false;
    }
}

function resetTime() {
    min_x = parseInt(document.getElementById('minx').value);
    max_x = parseInt(document.getElementById('maxx').value);
    min_y = parseInt(document.getElementById('miny').value);
    max_y = parseInt(document.getElementById('maxy').value);
    console.log('Resetting to (' + min_x + ', ' + min_y + ') (' + max_x + ', ' + max_y + ')');
    curr_simulation_step = starting_simulation_step;
}

function readSingleFile(e) {
    var fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', function(e) {
        curr_simulation_step = 1000000;
        pe_colors = {};
        state_colors = {};
        min_x = null; max_x = null;
        min_y = null; max_y = null;
        simulation_data = {};
        var offset = 0;
        var fileChunkSize = 32 * 1024 * 1024;

        var file = fileInput.files[0];
        var fileSize = file.size;
        var partialContents = '';

        var readEventHandler = function(evt) {
            if (evt.target.error == null) {
                var contents = partialContents + evt.target.result;
                console.log('read ' + evt.target.result.length + ' bytes.');
                var last_line_sep = contents.lastIndexOf(',,');
                if (last_line_sep === -1) {
                    console.log('No line separator found?');
                    return;
                }
                partialContents = contents.substring(last_line_sep + 2);
                contents = contents.substring(0, last_line_sep);
                offset += evt.target.result.length;
                if (offset >= fileSize) {
                    contents += partialContents;
                }

                var lines = contents.split(',,');
                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i].split(',');
                    var id = parseInt(line[0]);
                    var nfeatures = parseInt(line[1]);
                    var timestep = parseInt(line[2]);
                    var pe = parseInt(line[3]);

                    var features = {};
                    for (var j = 0; j < nfeatures; j++) {
                        features[parseInt(line[4 + 2 * j])] =
                            parseFloat(line[4 + 2 * j + 1]);
                    }

                    addPE(pe);
                    addState(features[2]);

                    var x = features[0];
                    var y = features[1];
                    if (min_x === null || x < min_x) min_x = x;
                    if (min_y === null || y < min_y) min_y = y;
                    if (max_x === null || x > max_x) max_x = x;
                    if (max_y === null || y > max_y) max_y = y;

                    if (!(timestep in simulation_data)) {
                        simulation_data[timestep] = {};
                    }
                    simulation_data[timestep][id] = new Particle(x, y, pe,
                            features[2]);

                    if (timestep < curr_simulation_step) {
                        curr_simulation_step = timestep;
                    }
                }

                if (offset < fileSize) {
                    chunkReaderBlock(offset, fileChunkSize, file);
                } else {
                    // Launch simulation
                    console.log('min = (' + min_x + ', ' + min_y + ') max = (' +
                                max_x + ', ' + max_y + ')');
                    console.log('PEs = ' + JSON.stringify(pe_colors));
                    console.log('States = ' + JSON.stringify(state_colors));

                    document.getElementById('minx').value = min_x;
                    document.getElementById('maxx').value = max_x;
                    document.getElementById('miny').value = min_y;
                    document.getElementById('maxy').value = max_y;

                    var curr_timestep = curr_simulation_step + 1;
                    while (curr_timestep in simulation_data) {
                        var particles = simulation_data[curr_timestep];
                        var last_particles = simulation_data[curr_timestep - 1];
                        for (var id in last_particles) {
                            if (!(id in particles)) {
                                particles[id] = last_particles[id];
                            }
                        }
                        curr_timestep += 1;
                    }

                    starting_simulation_step = curr_simulation_step;

                    // curr_timestep = curr_simulation_step;
                    // while (curr_timestep in simulation_data) {
                    //     console.log('Loaded ' + Object.keys(simulation_data[curr_timestep]).length + ' actors for timestep ' + curr_timestep);
                    //     curr_timestep += 1;
                    // }

                    loop();
                }
            } else {
                console.log("Read error: " + evt.target.error);
                return;
            }
        }

        chunkReaderBlock = function(_offset, length, _file) {
            var r = new FileReader();
            var blob = _file.slice(_offset, _offset + length);
            r.onload = readEventHandler;
            r.readAsText(blob);
        };

        chunkReaderBlock(offset, fileChunkSize, file);
    });
}

//Ading two methods
Particle.prototype.Draw = function (ctx) {
    if (color_based_on_state) {
        ctx.fillStyle = getStateColor(this.state);
    } else {
        ctx.fillStyle = getPEColor(this.pe);
    }

    var normalize_x = this.x - min_x;
    normalize_x = normalize_x / (max_x - min_x);
    normalize_x = normalize_x * canvas.width;

    var normalize_y = this.y - min_y;
    normalize_y = normalize_y / (max_y - min_y);
    normalize_y = normalize_y * canvas.height;

    ctx.fillRect(normalize_x, normalize_y, 2, 2);
}

function loop() {
    if (!(curr_simulation_step in simulation_data)) {
        curr_simulation_step = starting_simulation_step;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var particle_data = simulation_data[curr_simulation_step];
    // console.log('Drawing time ' + curr_simulation_step + ' w/ ' + Object.keys(particle_data).length + ' particles.');
    for (var key in particle_data) {
        particle_data[key].Draw(ctx);
    }

    document.getElementById("timestep").innerHTML = curr_simulation_step;
    curr_simulation_step += 1;

    requestAnimationFrame(loop);
}

readSingleFile();
