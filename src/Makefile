#####################################################################
# Copyright (c) 2005 Point Grey Research Inc.
#
# This Makefile is free software; Point Grey Research Inc. 
# gives unlimited permission to copy and/or distribute it,
# with or without modifications, as long as this notice is preserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, to the extent permitted by law; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.
#
#####################################################################
#PROFILEFLAGS = -pg -fprofile-arcs -ftest-coverage


# compilation flags
CPPFLAGS += -I. 
CPPFLAGS += -I../include
CPPFLAGS += -Wall -Wno-write-strings -g $(PROFILEFLAGS)

LDFLAGS += -L../lib $(PROFILEFLAGS)

LIBDIR = ../lib
INCDIR = ../include
INSTALLFLAGS += -p

BINDIR = ../bin

#EXECS		= $(EXEC1) $(EXEC2) $(EXEC3) $(EXEC4) $(EXEC5)
EXECS		= $(EXEC1) $(EXEC2) $(EXEC3) $(EXEC4) $(EXEC5) $(EXEC6) 

# library name and contents
NAME1		= image
LIB1		= lib$(NAME1).a
LIB1SRC		= $(NAME1).cpp	
INC1		= $(NAME1).h

NAME2		= logpolar
LIB2		= lib$(NAME2).a
LIB2SRC		= $(NAME2).cpp	
INC2		= $(NAME2).h

NAME3		= nn
LIB3		= lib$(NAME3).a
LIB3SRC		= $(NAME3).cpp	
INC3		= $(NAME3).h

#LIBS		= $(LIB1) 
LIBS 		+= -l$(NAME1)
LIBS 		+= -l$(NAME2)
LIBS 		+= -l$(NAME3)

# executable name and contents
EXEC1		= nn-bin
EXEC1SRC	= $(EXEC1).cpp
EXEC1DEP	= installlib installinclude

EXEC2		= im2logpol
EXEC2SRC	= $(EXEC2).cpp
EXEC2DEP	= installlib installinclude

ALLLIB        =  $(LIB1) $(LIB2) $(LIB3)
ALLINC         =  $(INC1) $(INC2) $(INC3)


all:	buildlibs installlib installinclude bin installbin

buildlibs: $(ALLLIB)

installbin: bin
	-install $(INSTALLFLAGS) $(EXECS) $(BINDIR)

installlib: buildlibs
	-install $(INSTALLFLAGS) $(ALLLIB) $(LIBDIR)

installinclude: 
	-install $(INSTALLFLAGS) $(ALLINC) $(INCDIR)

lib%.a: %.o
	ar -r $@ $^\

bin: $(EXECS)

$(EXEC1): $(EXEC1DEP) $(EXEC1SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $(EXEC1SRC:%.cpp=%.o) $(LIBS)

$(EXEC2): $(EXEC2SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(EXEC3): $(EXEC3SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(EXEC4): $(EXEC4SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(EXEC5): $(EXEC5SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(EXEC6): $(EXEC6SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(EXEC7): $(EXEC7SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(EXEC8): $(EXEC8SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(EXEC9): $(EXEC9SRC:%.cpp=%.o)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
	
%.o:%.cpp %.h
	g++ -c $(CXXFLAGS) $(CPPFLAGS) $*.cpp
clean:
	rm -f *~ *.o *.d *.a $(EXECS) 

#####################################################################
#
# $Id: Makefile,v 1.1 2009/12/17 01:53:19 mattg Exp $
# $Author: mattg $
# $Revision: 1.1 $
# $Date: 2009/12/17 01:53:19 $
#
#####################################################################

