#---------------------------------------------------------------------
#              
#------------------------------------------------------------------

CXX=g++


TARGET1 = ckastl_transform
TARGET2 = ckastl_min_element
TARGET3 = ckastl_merge
TARGET4 = ckastl_transform_grain
TARGET5 = ckastl_min_element2
TARGET6 = ckastl_min_element3
TARGET7 = ckastl_merge2
TARGET8 = ckastl_transform2

FILE1 = transform
FILE2 = min_element
FILE3 = merge
FILE4 = transform_grain
FILE5 = min_element2
FILE6 = min_element3
FILE7 = merge2
FILE8 = transform2

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) \
     $(TARGET7) $(TARGET8)

CKAAPI_CPPFLAGS= -I$(KAAPI_DIR)/include/
CKAAPI_LDFLAGS= -L$(KAAPI_DIR)/lib/ -lxkaapi -lxkaapi++ -lpthread

OFILES1=\
	transform.o

OFILES2=\
        min_element.o

OFILES3=\
        merge.o

OFILES4=\
	transform_grain.o

OFILES5=min_element2.o

OFILES6=min_element3.o

OFILES7=merge2.o

OFILES8=transform2.o

$(TARGET1):$(OFILES1)
	$(CXX) $(CFLAGS)  -o $(TARGET1) $(OFILES1) $(CKAAPI_LDFLAGS) 

$(TARGET2):$(OFILES2)
	$(CXX) $(CFLAGS)  -o $(TARGET2) $(OFILES2) $(CKAAPI_LDFLAGS)

$(TARGET3):$(OFILES3)
	$(CXX) $(CFLAGS)  -o $(TARGET3) $(OFILES3) $(CKAAPI_LDFLAGS)

$(TARGET4):$(OFILES4)
	$(CXX) $(CFLAGS)  -o $(TARGET4) $(OFILES4) $(CKAAPI_LDFLAGS) 

$(TARGET5):$(OFILES5)
	$(CXX) $(CFLAGS)  -o $(TARGET5) $(OFILES5) $(CKAAPI_LDFLAGS) 

$(TARGET6):$(OFILES6)
	$(CXX) $(CFLAGS)  -o $(TARGET6) $(OFILES6) $(CKAAPI_LDFLAGS) 

$(TARGET7):$(OFILES7)
	$(CXX) $(CFLAGS)  -o $(TARGET7) $(OFILES7) $(CKAAPI_LDFLAGS)

$(TARGET8):$(OFILES8)
	$(CXX) $(CFLAGS)  -o $(TARGET8) $(OFILES8) $(CKAAPI_LDFLAGS)

$(FILE1).o: $(FILE1).cpp $(FILE1).h
	$(CXX) $(CFLAGS) -c -o  $(FILE1).o $(CKAAPI_CPPFLAGS) $(FILE1).cpp

$(FILE2).o: $(FILE2).cpp $(FILE2).h
	$(CXX) $(CFLAGS) -c -o  $(FILE2).o $(CKAAPI_CPPFLAGS) $(FILE2).cpp

$(FILE3).o: $(FILE3).cpp $(FILE3).h
	$(CXX) $(CFLAGS) -c -o  $(FILE3).o $(CKAAPI_CPPFLAGS) $(FILE3).cpp

$(FILE4).o: $(FILE4).cpp $(FILE4).h
	$(CXX) $(CFLAGS) -c -o  $(FILE4).o $(CKAAPI_CPPFLAGS) $(FILE4).cpp

$(FILE5).o: $(FILE5).cpp $(FILE5).h
	$(CXX) $(CFLAGS) -c -o  $(FILE5).o $(CKAAPI_CPPFLAGS) $(FILE5).cpp

$(FILE6).o: $(FILE6).cpp $(FILE6).h
	$(CXX) $(CFLAGS) -c -o  $(FILE6).o $(CKAAPI_CPPFLAGS) $(FILE6).cpp

$(FILE7).o: $(FILE7).cpp $(FILE7).h
	$(CXX) $(CFLAGS) -c -o  $(FILE7).o $(CKAAPI_CPPFLAGS) $(FILE7).cpp

$(FILE8).o: $(FILE8).cpp $(FILE8).h
	$(CXX) $(CFLAGS) -c -o  $(FILE8).o $(CKAAPI_CPPFLAGS) $(FILE8).cpp


.PHONY : veryclean
.PHONY : clean

veryclean :
	rm  -f *.o $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) \
                   $(TARGET6) $(TARGET7) $(TARGET8)

clean :
	rm  -f *.o
