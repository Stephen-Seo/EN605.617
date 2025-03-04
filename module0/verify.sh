#!/bin/bash

echo "Starting sample CI verification script"

make helloworld

echo "Trying to execute ./helloworld"

OUTPUT=`./helloworld`
RETVAL=$?

if [ $RETVAL -eq 0 ]; then
  echo "Retval is 0, OK"
else
  echo "Retval is not 0, FAIL"
  make clean
  exit 1
fi

if [ "$OUTPUT" == "Hello World!" ]; then
  echo "Output is correct, OK"
else
  echo "Output is not right, FAIL"
  make clean
  exit 1
fi

make clean
