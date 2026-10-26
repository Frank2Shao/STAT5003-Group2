# Random Forest slides

Two slides using the group's SOLES theme, with simple English speaker notes.

Render from the repository root:

```sh
quarto render presentation/rf_model_slides.qmd
```

To merge into the group deck in the same presentation directory, insert:

```markdown
{{< include _rf-slides.qmd >}}
```

Keep `../figures/random_forest/confusion_matrix.png` available relative to the deck. No model training or additional R packages are needed to render these slides. The preview wrapper is not intended to replace the group deck.

Numbers come from the committed RF outputs. The table reports mean fold metrics; the figure pools out-of-fold predictions. These are validation results, not held-out test results. The selected-model/test-results section must be completed only after the group selects its final model.

AI assistance was used to draft slide text and speaker notes and assemble the Quarto files from saved project results. Please review the content and include this assistance in the group's disclosure as required.

Template attribution: SOLES reveal.js Quarto template by Januar Harianto, https://github.com/usyd-soles-edu/soles-revealjs. The repository retains its template licence and attribution.
