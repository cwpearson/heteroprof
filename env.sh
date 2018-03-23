#! /bin/bash
set -eou pipefail

# the time of the profile
NOW=`date +%Y%m%d-%H%M%S`

# where to look for cprof/profiler
if [ -z "${HETEROPROF_ROOT+xxx}" ]; then 
  export HETEROPROF_ROOT="$HOME/repos/heteroprof";
fi

# Check that libcprof.so exists
LIBHETEROPROF="$HETEROPROF_ROOT/lib/libheteroprof.so"
if [ ! -f "$LIBHETEROPROF" ]; then
    echo "$LIBHETEROPROF" "not found! try"
    echo "make -C $HETEROPROF_ROOT"
    exit -1
fi

## Control some profiling parameters.

# default output file
export CPROF_OUT="$NOW"_output.cprof
#export CPROF_ERR="err.cprof"

export CPROF_CUPTI_DEVICE_BUFFER_SIZE=1024

## Run the provided program. For example
#   ./env.sh examples/samples/vectorAdd/vec

if [ -z "${LD_PRELOAD+xxx}" ]; then 
  LD_PRELOAD="$LIBHETEROPROF" $@; # unset
else
  echo "Error: LD_PRELOAD is set before profile:"
  echo "\tLD_PRELOAD=$LD_PRELOAD"
fi
