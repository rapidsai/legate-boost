#ifndef __LEGATEBOOST_C_H__
#define __LEGATEBOOST_C_H__

enum LegateBoostOpCode {
  _OP_CODE_BASE = 0,
  BUILD_TREE    = 1,
  PREDICT       = 2,
  UPDATE_TREE   = 3,
  /* special */
  ERF = 4,
  LGAMMA = 5,
};

#endif  // __LEGATEBOOST_C_H__
