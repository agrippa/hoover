/** 
    Daniel Alabi
    Count-Min Sketch Implementation based on paper by
    Muthukrishnan and Cormode, 2004
**/

// define some constants
# define LONG_PRIME 4294967311l
# define MIN(a,b)  (a < b ? a : b)

/** CountMinSketch class definition here **/
class CountMinSketch {
  // width, depth 
  unsigned int w,d;
  
  // eps (for error), 0.01 < eps < 1
  // the smaller the better
  float eps;
  
  // gamma (probability for accuracy), 0 < gamma < 1
  // the bigger the better
  float gamma;
  
  // aj, bj \in Z_p
  // both elements of fild Z_p used in generation of hash
  // function
  unsigned int aj, bj;

  // total count so far
  unsigned int total; 

  // array of arrays of counters
  int **C;

  // array of hash values for a particular item 
  // contains two element arrays {aj,bj}
  unsigned **hashes;

  // generate "new" aj,bj
  void genajbj(unsigned **hashes, int i);

public:
  // constructor
  CountMinSketch(float eps, float gamma);

  CountMinSketch(const CountMinSketch &other);
  
  // update item (int) by count c
  void update(unsigned int item, int c);
  // update item (string) by count c
  void update(const char *item, int c);

  // estimate count of item i and return count
  unsigned int estimate(unsigned int item);
  unsigned int estimate(const char *item);

  // return total count
  unsigned int totalcount();

  // generates a hash value for a string
  // same as djb2 hash function
  unsigned int hashstr(const char *str);

  // destructor
  ~CountMinSketch();
};


