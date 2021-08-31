# HealthVer

This repository contains source code and the HealthVer dataset presented in the paper [Evidence-based Fact-Checking of Health-related Claims]() by Mourad Sarrouti, Asma Ben Abacha, Yassine Mrabet and Dina Demner-Fushman.

> The task of verifying the truthfulness of claims in textual documents, or fact-checking, has received significant attention in recent years. Many existing evidence-based fact-checking datasets contain synthetic claims and the models trained on these data might not be able to verify real-word claims. Particularly few studies addressed evidence-based fact-checking of health-related claims that require medical expertise or evidence from the scientific literature.
In this paper, we introduce HealthVer a new dataset for evidence-based fact-checking of health-related claims that allows to study the validity of real-world claims by evaluating their truthfulness against scientific articles. Using a three-step data creation method, we first retrieved real-world claims from snippets returned by a search engine for questions about COVID-19. Then we automatically retrieved and re-ranked relevant scientific papers using a T5 relevance-based model. Finally, the relations between each evidence statement and the associated claim were manually annotated as Support, Refute} and Neutral. To validate the created dataset of 14,330 evidence-claim pairs, we developed baseline models based on pretrained language models. Our experiments showed that training deep learning models on real-world medical claims greatly improves performance compared to models trained on synthetic and open-domain claims. Our results and manual analysis suggest that HealthVer provides a realistic and challenging dataset for future efforts on evidence-based fact-checking of health-related claims.

## Leaderboard

**UPDATE (September 2021)**:


## Citation

```bibtex
@inproceedings{Sarrouti2021Healthver,
  title={Evidence-based Fact-Checking of Health-related Claims},
  author={Mourad Sarrouti, Asma Ben Abacha, Yassine Mrabet and Dina Demner-Fushman},
  booktitle={EMNLP},
  year={2021},
}
```

## Contact

- Mourad Sarrouti, `sarrouti.mourad@gmail.com`
- Asma Ben Abacha, `asma.benabacha@nih.gov`
- Yassine Mrabet, `yassine.m'rabet@nih.gov`
- Dina Demner-Fushman, `ddemner@mail.nih.gov`
