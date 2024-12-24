# Honours Dissertation LaTeX Template

> Orginal HWU template: Pierre Le Bras
> Orginal ACM template: Association for Computing Machinery (ACM), used under [LaTeX Project Public License v1.3 (LPPL-1.3)](https://www.latex-project.org/lppl.txt)
> The ACM template was used as the styling basis and modified in `main.tex` to make it correspond with the Computer Science Department's dissertation format. 
> Authors: Pierre Le Bras & Usman Sanusi & Radu Mihailescu
> Heriot-Watt University, Computer Science Department
> Last update: September 2024

## Structure

The template project is structured as follows:
 - `main.tex` is the main file, pulling content from other files to create the full document for your submission. This is the default rendered file. **You should not modify anything in this file**.
 - `main_acm.tex` is an alternative main file, pulling content from other files to create a document with the ACM journal format. **You should not modify anything in this file**.
 - the `setup` directory contains `.tex` scripts used for loading packages, adjusting settings, defining macros etc. **You should not modify anything in this folder**.
 - the `acm_format` directory contains scripts ACM wrote for their journal submissions. **You should not modify anything in this folder**.
 - the `text` directory contains files with content for the document's main text. **These are the files you will edit**.
 - the `appendices` directory contains files with content for the document's appendices. **These files you may edit**.

### Setup

The main files will start by declaring the type of document and calling scripts in the `setup` folder. These will use variables declared in `text/variables.tex` to fill in some of the fields. The texts in `text/abstract.tex` and `text/acknowledgements.tex` are also loaded at this stage.

They then load the content of `text/body.tex`, which itself loads the content of the `text\main_body` folder.

Finally, it builds the list of references defined in `text/references.bib` and loads the content of `appendices\appendices.tex` (which loads other files in the `appendices` folder).

## Getting Started

 1. Create a copy of this project in overleaf: Menu -> Copy Project. (Alternatively, you can download the source and use them offline).
 2. Edit the variables to change the author name, title of dissertation, name of supervisor etc.
 3. Write the sections, and look at the sample sections to see how to include certain elements (figures, code listings, tables, cross-references, etc.)
    a. The five sample section should be a good starting point to structure the dissertation
    b. Make sure to edit `text/body.tex` if needed
    c. Add references to `text/references.bib`
 5. Complete the abstract, acknowledgements
 6. Write the appendices and edit `appendices/appendices.tex`
 8. Read through the document to check everything is in order

You can check `acm_format/sample_acmlarge.tex` for samples to include equations, tables, or images.

This template includes sample sections that contain examples of the following LaTeX features:
 - Lists, numbered and unnumbered
 - Tables
 - Figures, including having subfigures
 - Equations
 - Pseudo-code
 - Code listing (directly from code files)
 - Labeling and cross-referencing
 - Citations