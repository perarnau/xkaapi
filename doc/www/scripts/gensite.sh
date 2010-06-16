#!/bin/sh
set -e

SCRIPTDIR="$PWD"
TOP_SRCDIR="$PWD/../../trunk"
BUILDDIR="$PWD/.build"
DOC_SUBDIR="doc"
CONFIGURE_TARGET="--with-doc"
MAKE_TARGET="info doc"
TEMPLATE_FILE="template.html"
CSS_FILE="$SCRIPTDIR/info.css"
TITLE_SHORT=kaapi
TITLE_LONG="KAAPI Library Manual"
OUTDIR=website
GENERATED_DOCDIR="$DOC_SUBDIR/kaapi/kaapi/"
EXTRACTED_FILES="install.html compile_run.html atha.html tuto.html api.html"
IMGDIR=images

if test ! -d $TOP_SRCDIR/$DOC_SUBDIR; then
    echo "not in scripts directory" && exit 1
fi

#build dri
test -d .build || mkdir .build
cd $BUILDDIR
$TOP_SRCDIR/configure $CONFIGURE_TARGET
for target in $MAKE_TARGET; do
    make $target
done

cd $SCRIPTDIR

# prepare templates
sed -e  "/<!-- DOCS !-->/ r docs.input" \
    $TEMPLATE_FILE > template-doc.html
sed -e "/<!-- SECTION !-->/ r gendocs.input" \
    template-doc.html > gendocs_template
    

# gen all texinfo daoc
cd $TOP_SRCDIR/$DOC_SUBDIR
GENDOCS_TEMPLATE_DIR=$SCRIPTDIR $SCRIPTDIR/gendocs.sh --html "--css-include=$CSS_FILE" "$TITLE_SHORT" "$TITLE_LONG"
cd -

# create out dir
test -d $OUTDIR || mkdir $OUTDIR
test -d $OUTDIR/doc || mkdir $OUTDIR/doc

# copy generated doc
find $BUILDDIR/$GENERATED_DOCDIR -type f -exec cp '{}' $OUTDIR/doc/ \;
cp -r $TOP_SRCDIR/doc/manual/* $OUTDIR/doc/
cp *.html $OUTDIR/

# rework sections
TMPFILE=`mktemp`

for f in $EXTRACTED_FILES; do
    _EXTRACTED_FILES="$_EXTRACTED_FILES $OUTDIR/doc/$f"
done

./extract_section.sh 2 "Overview" < $OUTDIR/doc/kaapi.html > $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" \
    -e '/<!-- SECTION !-->/ i <div style="float: right; padding: 10px;"><img src="moais.jpg" alt="Picture" /></div>' \
    template.html > $OUTDIR/index.html

./extract_section.sh 3 "Get the source" < $OUTDIR/doc/kaapi.html > $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" \
    template.html > $OUTDIR/download.html

#./extract_section.sh 2 "Contacts" < website/doc/kaapi.html > $TMPFILE
#sed -e "/<!-- SECTION !-->/ r $TMPFILE" \
#    template.html > website/contact.html

./extract_section.sh 2 "Quick installation guide of KAAPI" < $OUTDIR/doc/kaapi.html > $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" template-doc.html > $OUTDIR/doc/install.html


./extract_section.sh 2 "Compile and Run Athapascan programs" < $OUTDIR/doc/kaapi.html > $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" template-doc.html > $OUTDIR/doc/compile_run.html

./extract_section.sh 2 "Athapascan Concept" < $OUTDIR/doc/kaapi.html > $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" template-doc.html > $OUTDIR/doc/atha.html

./extract_section.sh 2 "Athapascan tutorials" < $OUTDIR/doc/kaapi.html > $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" template-doc.html > $OUTDIR/doc/tuto.html

./extract_section.sh 2 "Athapascan Application Programming Interface" < $OUTDIR/doc/kaapi.html > $TMPFILE
sed -e "/<!-- SECTION !-->/ r $TMPFILE" template-doc.html > $OUTDIR/doc/api.html

for f in $_EXTRACTED_FILES; do
    grep -E 'href="#[^"]*' -o $f | cut -d '#' -f 2 | sed "1 d" |\
    while read r; do
        for o in $_EXTRACTED_FILES; do
            if grep -q "name=\"$r\"" $o; then
                F=`basename $o`
                sed -i -r -e "s@href=\"#$r@href=\"$F#$r@g" $f
            fi
        done
    done
done


rm -f $TMPFILE

# copy css and image files
cp *.css $OUTDIR/doc/
cp *.css $OUTDIR/

cp $IMGDIR/* $OUTDIR
cp $IMGDIR/* $OUTDIR/doc
cp $IMGDIR/* $OUTDIR/doc/html_node

cp -f $OUTDIR/doc/index.html $OUTDIR/doc/main.html

# help string 
echo
echo "***************************************************************************************"
echo "website built in \`./website/' repository"
echo "now type:"
echo 'rsync -av --exclude rss --exclude "*snapshot*" --delete website/ mylogin@shell.gforge.inria.fr:/home/groups/kaapi/htdocs'
echo "to publish the website from the \`website' repository"

#rm -rf .build template-doc.html gendocs_template
rm -rf template-doc.html gendocs_template
cd $TOP_SRCDIR/doc
rm -rf    *.pg\
      *.log\
      *.tp\
      *.toc\
      *.vr\
      *.aux\
      *.cp\
      *.ky\
      *.fns\
      manual\
      *.fn
