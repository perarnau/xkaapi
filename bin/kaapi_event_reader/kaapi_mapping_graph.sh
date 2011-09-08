#!/bin/sh


colorname[0]=cornflowerblue
colorname[1]=coral
colorname[2]=darkgoldenrod1
colorname[3]=darkolivegreen1
colorname[4]=darkorange
colorname[5]=darkorchid1
colorname[6]=darkolivegreen
colorname[7]=aquamarine3
colorname[8]=firebrick2
colorname[9]=gold1
colorname[10]=indigo
colorname[11]=lavender
colorname[12]=mediumvioletred
colorname[13]=royalblue1
colorname[14]=yellow1
colorname[15]=salmon


#
# Here graphfilename is the name of dot graph that describes the computation
#
graphfilename=$1
shift

#
# The $2 is the dirname of the set of event files that contains all the mappings.
# For instance, if $2= '/tmp/events.thierry', then the mapping files are
# /tmp/events.thierry.0.evt
# /tmp/events.thierry.1.evt
# ...
# until the last found filename /events.thierry.k.evt
# The number (0,1,...,k) represents the kid of the k-processor that has generated 
# the events.
mapfile=$1

# /tmp/events.thierry.0.evt
for (( k=0; k<100; ++k ))
do
  fileevent="$mapfile.$k.evt"
  if [ ! -e $fileevent ]
  then
    break;
  fi
  echo $fileevent
  listtask=`./bin/kaapi_event_reader -m $fileevent | cut -d\  -f2`
  for t in $listtask 
  do
    sed -e "s/\(.*\)$t\(.*\)color=orange/\1 $t\2color=${colorname[$k]}/" $graphfilename > /tmp/t
    mv /tmp/t $graphfilename
  done
done


