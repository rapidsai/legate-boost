#ifndef __LEGATEBOOST_C_H__
#define __LEGATEBOOST_C_H__

enum LegateBoostOpCode {  // NOLINT(performance-enum-size)
  _OP_CODE_BASE = 0,
  BUILD_TREE    = 1,
  PREDICT       = 2,
  UPDATE_TREE   = 3,
  /* special */
  ERF     = 4,
  LGAMMA  = 5,
  TGAMMA  = 6,
  DIGAMMA = 7,
  ZETA    = 8,
  /**/
  GATHER   = 9,
  RBF      = 10,
  BUILD_NN = 11,
};

#endif  // __LEGATEBOOST_C_H__
