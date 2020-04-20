import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))

def pdf(filein, fileout):
    #filein = "DatasaraPARTIES.csv"
    #fileout = "Data.pdf"
    txt = open(filein, 'r').read()
    docpdf = SimpleDocTemplate(fileout, pagesize = A4)
    style = getSampleStyleSheet()
    style.add(ParagraphStyle(name='Chinese',
                             fontName='STSong-Light',
                             fontSize=12,
                             leading=14,
                             wordWrap = 'CJK'))
    style.add(ParagraphStyle(name='Titre',
                             fontName='Courier',
                             fontSize=30,
                             leading=14,
                             wordWrap = 'CJK', 
                             textColor="#0040d9"))
    style.add(ParagraphStyle(name='Sous Titre',
                             fontName='Courier',
                             fontSize=25,
                             leading=14,
                             wordWrap = 'CJK', 
                             textColor="#0040d9"))
    story = []
    story.append(Paragraph("CHIFOUMI    INTELLIGENT", style["Titre"]))
    story.append(Spacer(0, cm * 1.3))
    story.append(Paragraph("\n\nJoueur : " + "Sara" + "\n\n\n", style["Sous Titre"]) )
    story.append(Spacer(0, cm * .8))
    paragraphs = txt.split("\n")
    for para in paragraphs:
        story.append(Paragraph(para, style["Chinese"]))
        story.append(Spacer(0, cm * .3))
    docpdf.build(story)

if __name__ == '__main__':
    filein = "DatasaraPARTIES.csv"
    fileout = "Data.pdf"
    pdf(filein, fileout)
    # if len(sys.argv) < 3:
    #     print('Utilisation : <script> textfile pdffile')
    #     sys.exit(1)
    # else :
    #     filein = sys.argv[1]
    #     fileout = sys.argv[2]
    #     pdf(filein)
