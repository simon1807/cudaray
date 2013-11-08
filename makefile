SELFDIR := $(dir $(lastword $(MAKEFILE_LIST)))

OUTPUT = /tmp/cudaray-build/cudaray
_OBJS = main.o cuda_main.o
LDFLAGS =

NVCC = nvcc
CC = gcc
CXX = g++
CFLAGS = -DDEBUG -D_BSD_SOURCE -Wall -Wextra -Wno-unused-parameter -pipe -g3 -O0
NVCFLAGS = -arch=sm_20 -O0
LDFLAGS += `OcelotConfig -l`
BUILDIR = $(shell dirname $(OUTPUT))
OBJS = $(patsubst %,$(BUILDIR)/%,$(_OBJS))

ifeq ($(ENABLE_GHETTO_CUDA), 1)
    NVCC = $(CXX)
    NVCFLAGS = $(CFLAGS) -x c++
endif

$(OUTPUT): $(OBJS)
	@echo "[LD] `basename $@`"
	@$(CXX) $(OBJS) $(CFLAGS) $(LDFLAGS) -o $(OUTPUT)

-include $(OBJS:.o=.d)

$(BUILDIR)/%.o: %.c
	@mkdir -p $(BUILDIR)
	@mkdir -p $(shell dirname $(@))
	@echo "[CC] `basename $@`"
	@$(CC) -c $(CFLAGS) $*.c -o $@
	@echo "[MM] `basename $(@:.o=.d)`"
	@$(CC) -MM $(CFLAGS) $*.c > $(@:.o=.d)
	@mv -f $(@:.o=.d) $(@:.o=.d).tmp
	@sed -e 's|.*:|$*.o:|' < $(@:.o=.d).tmp > $(@:.o=.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:.o=.d).tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $(@:.o=.d)
	@rm -f $(@:.o=.d).tmp

$(BUILDIR)/%.o: %.cpp
	@mkdir -p $(BUILDIR)
	@mkdir -p $(shell dirname $(@))
	@echo "[CXX] `basename $@`"
	@$(CXX) -c $(CFLAGS) $*.cpp -o $@
	@echo "[MM] `basename $(@:.o=.d)`"
	@$(CXX) -MM $(CFLAGS) $*.cpp > $(@:.o=.d)
	@mv -f $(@:.o=.d) $(@:.o=.d).tmp
	@sed -e 's|.*:|$*.o:|' < $(@:.o=.d).tmp > $(@:.o=.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:.o=.d).tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $(@:.o=.d)
	@rm -f $(@:.o=.d).tmp

$(BUILDIR)/%.o: %.cu
	@mkdir -p $(BUILDIR)
	@mkdir -p $(shell dirname $(@))
	@echo "[NVCC] `basename $@`"
	@$(NVCC) -c $(NVCFLAGS) $*.cu -o $@
	@echo "[MM] `basename $(@:.o=.d)`"
	@$(CC) -MM $(CFLAGS) -x c++ $*.cu > $(@:.o=.d)
	@mv -f $(@:.o=.d) $(@:.o=.d).tmp
	@sed -e 's|.*:|$*.o:|' < $(@:.o=.d).tmp > $(@:.o=.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:.o=.d).tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $(@:.o=.d)
	@rm -f $(@:.o=.d).tmp

clean:
	@echo "[CLEAN]"
	@find $(BUILDIR) -type f -name "*.o" | xargs --no-run-if-empty rm
	@find $(BUILDIR) -type f -name "*.d" | xargs --no-run-if-empty rm
	@rm -f $(OUTPUT)

run: $(OUTPUT)
	@$(OUTPUT)

all: $(OUTPUT)
