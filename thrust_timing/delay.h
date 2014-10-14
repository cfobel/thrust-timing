#ifndef ___SORT_TIMING__DELAY__H___
#define ___SORT_TIMING__DELAY__H___


template <typename DelayIterator>
struct delay {
  typedef float result_type;
  DelayIterator delays;
  int32_t nrows;
  int32_t ncols;

  delay(DelayIterator delays, int32_t nrows, int32_t ncols)
    : delays(delays), nrows(nrows), ncols(ncols) {}

  template <typename T1, typename T2, typename T3>
  result_type operator() (T1 delay_type, T2 delta_x, T3 delta_y) {
    size_t stride = 0;
    size_t offset = 0;

    if (delta_x == 0 && delta_y == 0) {
      return 0;
    } else if (delay_type == 0) {
      /* Delay is logic-to-logic. */
      stride = ncols;
      offset = 0;
    } else if (delay_type == 20) {
      /* Delay is logic-to-io. */
      stride = ncols + 1;
      offset = nrows * ncols;
    } else if (delay_type == 1) {
      /* Delay is io-to-io. */
      stride = ncols + 1;
      offset = nrows * ncols + (nrows + 1) * (ncols + 1);
    } else if (delay_type == 21) {
      /* Delay is io-to-io. */
      stride = ncols + 2;
      offset = nrows * ncols + 2 * (nrows + 1) * (ncols + 1);
    }
    return *(delays + offset + stride * delta_x + delta_y);
  }
};


template <typename T>
struct connection_criticality {
  typedef float result_type;
  float critical_path;

  connection_criticality(T critical_path) : critical_path(critical_path) {}

  template <typename T1, typename T2, typename T3>
  result_type operator() (T1 arrival_time, T2 delay, T3 departure_time) {
    return (arrival_time + delay + departure_time) / critical_path;
  }
};


template <typename T>
struct connection_cost {
  typedef float result_type;
  float critical_path;
  float criticality_exp;

  connection_cost(T critical_path, T criticality_exp)
    : critical_path(critical_path), criticality_exp(criticality_exp) {}

  template <typename T1, typename T2, typename T3>
  result_type operator() (T1 arrival_time, T2 delay, T3 departure_time) {
    return pow((arrival_time + delay + departure_time) / critical_path,
               criticality_exp) * delay;
  }
};


#endif  // #ifndef ___SORT_TIMING__DELAY__H___
