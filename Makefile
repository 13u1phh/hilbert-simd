VENV=venv
PYTHON=${VENV}/bin/python3

.PHONY: hilbert_test hilbert_bm run_bm clean

bin/hilbert_test: hilbert_test.cc hilbert.h luts.h gen_vec_macros.py make_luts.py hilbert_ref_impl.cc hilbert_ref_impl.h 
	mkdir -p bin
	${PYTHON} gen_vec_macros.py > luts.h
	g++ hilbert_test.cc hilbert_ref_impl.cc --std=c++20 -mavx512vbmi -g -o bin/hilbert_test

hilbert_test: bin/hilbert_test

BMDIR=benchmark
BMBUILD=${BMDIR}/build
BMLIB=${BMBUILD}/src/libbenchmark.a

${BMDIR}/.git:
	git submodule update --init --recursive

${BMLIB}: ${BMDIR}/.git
	cd benchmark && \
	cmake -E make_directory "build" && \
	cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -S . -B "build" && \
	cmake --build "build" --config Release

bin/hilbert_bm: hilbert_bm.cc hilbert.h hilbert_256_manual.h hilbert_16.h hilbert_cs.h luts.h gen_vec_macros.py make_luts.py ${BMLIB} 
	mkdir -p bin
	${PYTHON} gen_vec_macros.py > luts.h
	g++ hilbert_bm.cc -isystem ${BMDIR}/include ${BMLIB} -lpthread --std=c++20 -mavx512vbmi -O3 -o bin/hilbert_bm

hilbert_bm: bin/hilbert_bm

run_bm: hilbert_bm
	taskset -c 1 ./bin/hilbert_bm

clean:
	rm -f bin/*

