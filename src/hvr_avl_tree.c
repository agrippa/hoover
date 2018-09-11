/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "hvr_avl_tree.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/* Helper function that allocates a new node with the given key and
    NULL left and right pointers. */
static hvr_avl_tree_node_t *newNode(hvr_vertex_id_t key) {
    hvr_avl_tree_node_t* node = (hvr_avl_tree_node_t*)malloc(
            sizeof(hvr_avl_tree_node_t));
    assert(node);
    node->key = key;
    node->subtree = NULL;
    node->linearized = NULL;
    node->left = NULL;
    node->right = NULL;
    node->height = 1;  // new node is initially added at leaf
    return(node);
}

// A utility function to get height of the tree
static int height(hvr_avl_tree_node_t *N)
{
    if (N == NULL)
        return 0;
    return N->height;
}
 
// A utility function to right rotate subtree rooted with y
// See the diagram given above.
static hvr_avl_tree_node_t *rightRotate(hvr_avl_tree_node_t *y)
{
    hvr_avl_tree_node_t *x = y->left;
    hvr_avl_tree_node_t *T2 = x->right;

    // Perform rotation
    x->right = y;
    y->left = T2;

    // Update heights
    y->height = MAX(height(y->left), height(y->right))+1;
    x->height = MAX(height(x->left), height(x->right))+1;

    // Return new root
    return x;
}
 
// A utility function to left rotate subtree rooted with x
// See the diagram given above.
static hvr_avl_tree_node_t *leftRotate(hvr_avl_tree_node_t *x)
{
    hvr_avl_tree_node_t *y = x->right;
    hvr_avl_tree_node_t *T2 = y->left;

    // Perform rotation
    y->left = x;
    x->right = T2;

    //  Update heights
    x->height = MAX(height(x->left), height(x->right))+1;
    y->height = MAX(height(y->left), height(y->right))+1;

    // Return new root
    return y;
}
 
// Get Balance factor of node N
static int getBalance(hvr_avl_tree_node_t *N)
{
    if (N == NULL)
        return 0;
    return height(N->left) - height(N->right);
}
 
hvr_avl_tree_node_t* hvr_tree_insert(hvr_avl_tree_node_t* node,
        hvr_vertex_id_t key) {
    /* 1.  Perform the normal BST rotation */
    if (node == NULL)
        return(newNode(key));

    if (key < node->key)
        node->left  = hvr_tree_insert(node->left, key);
    else if (key > node->key)
        node->right = hvr_tree_insert(node->right, key);
    else // Equal keys not allowed
        return node;

    /* 2. Update height of this ancestor node */
    node->height = 1 + MAX(height(node->left),
            height(node->right));

    /* 3. Get the balance factor of this ancestor
       node to check whether this node became
       unbalanced */
    int balance = getBalance(node);

    // If this node becomes unbalanced, then there are 4 cases

    // Left Left Case
    if (balance > 1 && key < node->left->key)
        return rightRotate(node);

    // Right Right Case
    if (balance < -1 && key > node->right->key)
        return leftRotate(node);

    // Left Right Case
    if (balance > 1 && key > node->left->key)
    {
        node->left =  leftRotate(node->left);
        return rightRotate(node);
    }

    // Right Left Case
    if (balance < -1 && key < node->right->key)
    {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    /* return the (unchanged) node pointer */
    return node;
}
 
/* Given a non-empty binary search tree, return the
      node with minimum key value found in that tree.
         Note that the entire tree does not need to be
            searched. */
hvr_avl_tree_node_t * minValueNode(hvr_avl_tree_node_t* node)
{
    hvr_avl_tree_node_t* current = node;

    /* loop down to find the leftmost leaf */
    while (current->left != NULL)
        current = current->left;

    return current;
}
 
// Recursive function to delete a node with given key
// from subtree with given root. It returns root of
// the modified subtree.
hvr_avl_tree_node_t* hvr_tree_remove(hvr_avl_tree_node_t *root,
        hvr_vertex_id_t key) {
    // STEP 1: PERFORM STANDARD BST DELETE

    if (root == NULL)
        return root;

    // If the key to be deleted is smaller than the
    // root's key, then it lies in left subtree
    if ( key < root->key )
        root->left = hvr_tree_remove(root->left, key);

    // If the key to be deleted is greater than the
    // root's key, then it lies in right subtree
    else if( key > root->key )
        root->right = hvr_tree_remove(root->right, key);

    // if key is same as root's key, then This is
    // the node to be deleted
    else
    {
        // node with only one child or no child
        if( (root->left == NULL) || (root->right == NULL) )
        {
            hvr_avl_tree_node_t *temp = root->left ? root->left :
                root->right;

            // No child case
            if (temp == NULL)
            {
                temp = root;
                root = NULL;
            }
            else {
                // One child case
                memcpy(root, temp, sizeof(*root));
            }
            // the non-empty child
            free(temp);
        }
        else
        {
            // node with two children: Get the inorder
            // successor (smallest in the right subtree)
            hvr_avl_tree_node_t* temp = minValueNode(root->right);

            // Copy the inorder successor's data to this node
            root->key = temp->key;

            // Delete the inorder successor
            root->right = hvr_tree_remove(root->right, temp->key);
        }
    }

    // If the tree had only one node then return
    if (root == NULL)
        return root;

    // STEP 2: UPDATE HEIGHT OF THE CURRENT NODE
    root->height = 1 + MAX(height(root->left),
            height(root->right));

    // STEP 3: GET THE BALANCE FACTOR OF THIS NODE (to
    // check whether this node became unbalanced)
    int balance = getBalance(root);

    // If this node becomes unbalanced, then there are 4 cases

    // Left Left Case
    if (balance > 1 && getBalance(root->left) >= 0)
        return rightRotate(root);

    // Left Right Case
    if (balance > 1 && getBalance(root->left) < 0)
    {
        root->left =  leftRotate(root->left);
        return rightRotate(root);
    }

    // Right Right Case
    if (balance < -1 && getBalance(root->right) <= 0)
        return leftRotate(root);

    // Right Left Case
    if (balance < -1 && getBalance(root->right) > 0)
    {
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }

    return root;
}

static hvr_avl_tree_node_t *hvr_tree_find_helper(hvr_avl_tree_node_t *curr,
        hvr_vertex_id_t key, hvr_avl_tree_node_t **parent) {
    if (curr == NULL) {
        return NULL;
    }

    if (key < curr->key) {
        *parent = curr;
        return hvr_tree_find_helper(curr->left, key, parent);
    } else if (key > curr->key) {
        *parent = curr;
        return hvr_tree_find_helper(curr->right, key, parent);
    } else {
        return curr;
    }
}

hvr_avl_tree_node_t *hvr_tree_find(hvr_avl_tree_node_t *curr,
        hvr_vertex_id_t key) {
    hvr_avl_tree_node_t *unused;
    return hvr_tree_find_helper(curr, key, &unused);
}

void hvr_tree_destroy(hvr_avl_tree_node_t *curr) {
    if (curr == NULL) {
        return;
    }

    hvr_tree_destroy(curr->left);
    hvr_tree_destroy(curr->right);
    hvr_tree_destroy(curr->subtree);
    if (curr->linearized) free(curr->linearized);
    free(curr);
}

size_t hvr_tree_size(hvr_avl_tree_node_t *curr) {
    if (curr == NULL) {
        return 0;
    }
    return 1 + hvr_tree_size(curr->left) + hvr_tree_size(curr->right);
}

static void hvr_tree_linearize_helper(hvr_vertex_id_t *arr, unsigned *index,
        hvr_avl_tree_node_t *curr) {
    if (curr == NULL) {
        return;
    }

    hvr_tree_linearize_helper(arr, index, curr->left);
    hvr_tree_linearize_helper(arr, index, curr->right);
    arr[*index] = curr->key;
    *index += 1;
}

size_t hvr_tree_linearize(hvr_vertex_id_t **arr, hvr_avl_tree_node_t *curr) {
    if (curr->linearized == NULL) {
        const size_t tree_size = hvr_tree_size(curr);
        hvr_vertex_id_t *linearized = (hvr_vertex_id_t *)malloc(
                tree_size * sizeof(*linearized));
        assert(linearized);

        unsigned index = 0;
        hvr_tree_linearize_helper(linearized, &index, curr);
        assert(index == tree_size);

        curr->linearized = linearized;
        curr->linearized_length = tree_size;
    }

    *arr = curr->linearized;

    return curr->linearized_length;
}
