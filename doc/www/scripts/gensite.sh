#!/usr/bin/env sh

#
SCRIPTDIR="$PWD"
BUILDDIR="$PWD/.build"
OUTDIR="$BUILDDIR/www"

# make builddir
[ -d $BUILDDIR ] || mkdir $BUILDDIR

# make outdir
[ -d $OUTDIR ] || mkdir $OUTDIR
[ -d $OUTDIR/doc ] || mkdir $OUTDIR/doc

# prepare templates
TEMPLATE_FILE="template.html"
sed -e  "/<!-- DOCS !-->/ r docs.input" \
    $TEMPLATE_FILE > template-doc.html
sed -e "/<!-- SECTION !-->/ r gendocs.input" \
    template-doc.html > gendocs_template

# generate Manual.html
MANUALDIR=$BUILDDIR/Manual
[ -d $MANUALDIR ] || mkdir $MANUALDIR
cd ../..
latex2html -split +0 -dir $MANUALDIR Manual.tex
cd $SCRIPTDIR

# generate section list
SECTIONS=""
SECTIONS="$SECTIONS Introduction"
SECTIONS="$SECTIONS Installation"
SECTIONS="$SECTIONS Programming with X-Kaapi"
SECTIONS="$SECTIONS Running X-Kaapi"

# extract sections
MANUALFILE=$MANUALDIR/Manual.html
TMPFILE=`mktemp`

# index.html
./extract_section.py "Overview" $MANUALFILE $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" \
    -e '/<!-- SECTION !-->/ i <div style="float: right; padding: 10px;"><img src="moais.jpg" alt="Picture" /></div>' \
    template.html > $OUTDIR/index.html

# install.html
./extract_section.py "Installing" $MANUALFILE $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" template-doc.html > $OUTDIR/doc/install.html

# program.html
./extract_section.py "Programming" $MANUALFILE $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" template-doc.html > $OUTDIR/doc/program.html

# run.html
./extract_section.py "Running" $MANUALFILE $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" template-doc.html > $OUTDIR/doc/run.html

# replace path components
XFILES="install.html program.html run.html"
_XFILES=""
for f in $XFILES; do
    _XFILES="$_XFILES $OUTDIR/doc/$f"
done
for f in $_XFILES; do
    grep -E 'href="#[^"]*' -o $f | cut -d '#' -f 2 | sed "1 d" |\
    while read r; do
        for o in $_XFILES; do
            if grep -q "name=\"$r\"" $o; then
                F=`basename $o`
                sed -i -r -e "s@href=\"#$r@href=\"$F#$r@g" $f
            fi
        done
    done
done

rm $TMPFILE

# copy css and image files
IMGDIR=./images
cp *.css $OUTDIR/
cp $IMGDIR/* $OUTDIR
cp *.css $OUTDIR/doc/
cp $IMGDIR/* $OUTDIR/doc
#cp $IMGDIR/* $OUTDIR/doc/html_node
#cp -f $OUTDIR/doc/index.html $OUTDIR/doc/main.html
