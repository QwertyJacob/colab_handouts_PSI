# Colab_handouts_PSI
Dispense Google Colab, corso PSI-Varese, Università degli Studi dell'Insubria.

Contiene parti che sono traduzione e addattamento di:

- Probability for Computer Science, D. Forsyth, Springer Nature, 2018
- Probability for Computer Science (CS109 Course Reader), C. Piech, Standford, 2024
- Probability and Statistics for Computer Scientists, M. Baron, CRC Press, 2014

# Come utilizzare

Per aprire uno di questi notebook in colab:
> Basta andare su https://colab.research.google.com/github/QwertyJacob/colab_handouts_PSI/blob/main/[NOME_DEL_FILE].ipynb

Per esempio, per il primo file:

> https://colab.research.google.com/github/QwertyJacob/colab_handouts_PSI/blob/main/0_incertezza.ipynb

## Se vuoi aprire su Binder:

> https://mybinder.org/v2/gh/QwertyJacob/colab_handouts_PSI/HEAD

## Se vuoi aprire in Jupyter Notebook locale:

- assicurati di avere installato [jupyter](https://jupyter.org/install)
- apri il file [NOME_DEL_FILE].ipynb con jupyter notebook

Per usare facilmente la liberira [manim](https://www.manim.community), è meglio usare il [Docker](https://www.docker.com/) container con le librerie già installate:

- Dopo installare Docker, esegui il build dell'immagine e fai partire il container con il comando:
```bash
source run_local.sh
```
Dopodiché puoi usare jupyter notebook in locale su http://localhost:8888/ oppure aprire VsCode nel container con l'estensione [Container Tools](https://code.visualstudio.com/docs/devcontainers/containers).

# Come contribuire
1. Premere _Fork_ in alto a destra
2. Selezionare il link del fork che appare nella pagina
3. Aprire il documento su cui si vuole contribuire e premere su _Open in Colab_ (in alto a sinistra, all'inizio del documento)
4. Una volta modificato su Colab, premere File -> Salva una copia su _GitHub_ -> OK
5. Tornare sulla propria pagina _GitHub_ del fork
6. Premere su _Contribute_
7. Premere _Open pull request_
8. Scrivere una descrizione delle modifiche e/o aggiunte apportate
9. Premere su _Create pull request_
10. Le modifiche verranno poi accorpate dai responsabili della repo ;)
11. *Tieni il tuo fork sincronizzato!*

Feel free to contribute! Other students will appreciate!
