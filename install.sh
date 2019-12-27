set -e  # exit when any command fails
set -x

# Note that this needs Git intalled, so we check for that.

git --version 2>&1 >/dev/null
GIT_IS_AVAILABLE=$?
if [ $GIT_IS_AVAILABLE -ne 0 ]; then #...
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    sudo apt-get install git
  elif [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
    brew install git
  else
    echo "The OS '$OSTYPE' is not supported (Only Linux and MacOS is). " \
         "Feel free to contribute the install for a new OS."
    exit 1
  fi
fi

if [ ! -d "pybind11" ]; then
    git clone -b 'v2.2.4' --single-branch --depth 1 https://github.com/pybind/pybind11.git
fi

if [ ! -d "handIndex/hand-isomorphism"]; then
    git clone -b 'master' --single-branch --depth 1 https://github.com/kdub0/hand-isomorphism.git
fi

if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake ..
make -j8
cp handIndex/*.so ../
cp handRank/*.so ../
