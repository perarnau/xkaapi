INCLUDE(UseLATEX.cmake)
ADD_LATEX_DOCUMENT(specs.tex
  INPUTS mutex.tex
  IMAGES stack.pdf
)

# # 
# # Find LaTeX
# #
# FIND_PACKAGE(LATEX)

# set (STY "dot2texi.sty")
# set (FIG "stack.pdf")
# set (LATEX_COMPILER "pdflatex")

# CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/${STY}" "${CMAKE_CURRENT_BINARY_DIR}/${STY}" COPYONLY)
# CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/${FIG}" "${CMAKE_CURRENT_BINARY_DIR}/${FIG}" COPYONLY)

# IF(LATEX_COMPILER)
#   ADD_CUSTOM_COMMAND( 
#     OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/specs.pdf
#     COMMAND   ${LATEX_COMPILER}
#     ARGS      ${CMAKE_CURRENT_SOURCE_DIR}/specs.tex
#     DEPENDS   ${CMAKE_CURRENT_SOURCE_DIR}/specs.tex
#     COMMENT   "Tex2dvi"
#   )
# #  IF(DVIPS_CONVERTER)
# #    ADD_CUSTOM_COMMAND( 
# #      OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/specs.ps
# #      COMMAND   ${DVIPS_CONVERTER}
# #      ARGS      ${CMAKE_CURRENT_BINARY_DIR}/specs.dvi
# #                -o ${CMAKE_CURRENT_BINARY_DIR}/specs.ps
# #      DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/specs.dvi
# #      COMMENT   "dvi2ps"
# #   )
# #
# # IF(PS2PDF_CONVERTER)
# #    ADD_CUSTOM_COMMAND( 
# #      OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/specs.pdf
# #      COMMAND   ${PS2PDF_CONVERTER}
# #      ARGS      ${CMAKE_CURRENT_BINARY_DIR}/specs.ps
# #      DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/specs.ps
# #      COMMENT   "ps2pdf"
# #    )
#     ADD_CUSTOM_TARGET(LaTeXDocument ALL echo
#       DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/specs.pdf
#     )
# #    ENDIF(PS2PDF_CONVERTER)
# #  ENDIF(DVIPS_CONVERTER)
# ENDIF(LATEX_COMPILER)
