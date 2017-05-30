#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "LRSpline/LRSplineSurface.h"

using namespace std;
using namespace LR;


// check that new knots contain all original knots
static bool check_orig_knots_is_subset(const std::vector<double>& knots, const std::vector<double>& new_knots) {
  for (auto& it : new_knots)
    if (std::find_if(knots.begin(), knots.end(), [it](double a){return fabs(it-a) < 1e-4;}) == knots.end())
      return false;

  return true;
}


// 2 patch, flip + reverse
static int case1() {
  // create objects
  LRSplineSurface lr1(5,5,4,4);
  LRSplineSurface lr2(5,5,4,4);

  // do some refinements
  lr1.refineElement(0);
  lr1.refineElement(4);
  lr1.refineElement(6);
  lr1.refineElement(4);
  vector<double> knots = lr1.getEdgeKnots(WEST, true);
  cout << "Edge nodes on WEST side:\n";
  for(auto d : knots)
    cout << d << endl;
  // manually reverse knots (0,1) -> (1,0)
  for(int i=0; i<knots.size(); i++)
    knots[i] = 1-knots[i];

  // match on other side
  lr2.matchParametricEdge(NORTH, knots, true);
  cout << lr2 << endl;

  auto new_knots = lr2.getEdgeKnots(NORTH, true);
  if (!check_orig_knots_is_subset(knots, new_knots))
    return 1;

  // dump files for easy inspection
  ofstream out1("lr1.eps");
  lr1.writePostscriptMesh(out1);
  ofstream out2("lr2.eps");
  lr2.writePostscriptMesh(out2);
  out1.close();
  out2.close();

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Need a parameter - the test to run" << std::endl;
    std::cerr << "1 = 2 patch, flip + reverse" << std::endl;
    return 1;
  }

  if (atoi(argv[1]) == 1)
    return case1();

  return 1;
} 
