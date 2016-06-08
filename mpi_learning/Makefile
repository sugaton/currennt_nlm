IFLAGS = -I/home/usr8/14M53551/local/include -I/usr/apps.sp3/cuda/7.0/include -I/usr/apps.sp3/mpi/openmpi/1.8.2/g4.3.4_cuda7.0/include -I/mpi
LFLAGS = -L/home/usr8/14M53551/local/lib -L/home/usr8/14M53551/local/lib64 -L/usr/apps.sp3/mpi/openmpi/1.8.2/g4.3.4_cuda7.0/lib -L/usr/apps.sp3/cuda/7.0/lib64
LLIBS = -lboost_mpi -lboost_serialization -pthread -lmpi_cxx -lmpi -lboost_program_options -lboost_system -lboost_filesystem -lboost_thread -lboost_random -lcudnn -lcudart -lcublas

GXX = g++ -std=c++11 -O3 -D_MYMPI $(IFLAGS) $(LFLAGS) $(LLIBS)
NVCC = nvcc -O3 -std=c++11 -arch=sm_35 -DNEWSOFTMAX -D_MYMPI $(IFLAGS) $(LFLAGS) $(LLIBS)
MPICXX = mpic++ -O3 -std=c++11 $(IFLAGS) $(LFLAGS) $(LLIBS)

clibD = ../currennt_lib/src
OUTDIR = build/tmp

srcCxx := $(wildcard $(clibD)/corpus/*.cpp) \
          $(wildcard $(clibD)/data_sets/*.cpp) \
          $(wildcard $(clibD)/helpers/*.cpp) \
          $(wildcard $(clibD)/layers/*.cpp) \
          $(wildcard $(clibD)/rnnlm/*.cpp) \
          $(wildcard $(clibD)/*.cpp) \
          main_nlm.cpp

srcCu := $(wildcard $(clibD)/helpers/*.cu) \
         $(wildcard $(clibD)/layers/*.cu) \
         $(wildcard $(clibD)/lmOptimizers/*.cu) \
         $(wildcard $(clibD)/rnnlm/*.cu) \
         $(wildcard $(clibD)/*.cu) 

OBJS := $(srcCxx:%.cpp=$(OUTDIR)/%.o) $(srcCu:%.cu=$(OUTDIR)/%.o)
DEPS := $(srcCxx:%.cpp=$(OUTDIR)/%.d) $(srcCu:%.cu=$(OUTDIR)/%.d)

main := currennt_nlm 


#
all:  $(main)


-include $(DEPS)

$(main) : $(OBJS)
	$(MPICXX) -o $@ $^ 

$(OUTDIR)/%.o: %.cpp
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(GXX) -c -MMD -MP -MF $(@:%.o=%.d) -o $@ $< 

$(OUTDIR)/%.o: %.cu
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(NVCC) -M $(@:$(OUTDIR)/%.o=%.cu) > $(@:%.o=%.d)
	$(NVCC) -c -o $@ $<
