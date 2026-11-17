# Airline Review Sentiment Analysis

**Python | Pandas | Matplotlib | NLP | Exploratory visualisation | Evaluation**

A five-person Singapore Institute of Technology academic team project for **INF1002: Programming Fundamentals**, Team P4-4. The project processes airline reviews, applies AFINN lexicon-based sentiment analysis, and explores the results through charts and a Flask review interface.

The analytical question is how sentiment expressed in the collected reviews compares across airlines and over time, and how a ranking derived from those reviews differs from the external ranking recorded in the project.

## My contribution

I am **Daryl Lai Weng Hong**, GitHub [StocialAbyss](https://github.com/StocialAbyss). My recorded contributions include:

- Adding and updating `visualisations.py`, using Pandas to aggregate results and Matplotlib to produce comparative charts.
- Adding saved ranking-comparison charts, sentiment-distribution boxplots and monthly sentiment trends.
- Adding evaluation calculations and exported accuracy, precision, recall and F1 results using rating-derived reference labels.

For traceable examples, see my [visualisation implementation commit](https://github.com/DyanelLancea/Airline-Review-Sentiment-Analysis/commit/03ca83d07f25b4f99ac989c4b87991c6033e6890) and [follow-up changes](https://github.com/DyanelLancea/Airline-Review-Sentiment-Analysis/commit/227baeadde0aa1c0cb8fdccbbaea2c0134d73662).

The sentiment pipeline, data cleaning, web interface and other algorithms are shared project components. This portfolio does not claim that I independently built the entire application.

## Selected visualisations

### External ranking versus sentiment-derived ranking

![Horizontal bars comparing the recorded external airline ranking with the ranking derived from average review sentiment](visualisations/Official_vs_Sentiment.png)

The grouped bars make disagreements between the two rankings visible. For example, Hainan Airlines is ranked tenth in the recorded external list but first by average sentiment in this review sample. This is a comparison of different measures and samples, not evidence that either ranking is incorrect or that one airline is objectively best. Lower rank numbers are better.

### Sentiment distributions by airline

![Boxplots showing the distribution of normalised review sentiment for each airline](visualisations/boxplot_sentiment_consistency.png)

The boxplots show the median and interquartile range, with green triangles marking means. They reveal variation that would be hidden by reporting only an average. Points outside the whiskers are statistical outliers, not automatically invalid reviews.

### Monthly sentiment trends

![Monthly average sentiment with Singapore Airlines highlighted and other airlines shown faintly](visualisations/trend_spotlight.png)

The chart highlights Singapore Airlines against the other airlines using a rolling average of monthly sentiment means. Interpretation should consider review counts, missing months and changes in the review sample. The code uses a three-row rolling window on the monthly index; a strict three-calendar-month interpretation requires checking for gaps.

## How the team project works

1. Collect and prepare airline review data.
2. Score review text using the AFINN sentiment lexicon and normalise scores by text length.
3. Aggregate results by airline and time period for visual comparison.
4. Compare predicted polarity with positive/negative labels derived from review ratings.
5. Present review results through the Flask interface and export analysis outputs.

This is lexicon-based NLP, not generative AI or a newly trained language model.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| `visualisations.py` | Chart generation, evaluation calculations and keyword-summary output. |
| `visualisations/` | Saved charts, evaluation CSV and keyword summaries. |
| `sentiment_analysis.py` | Shared preprocessing and sentiment-analysis functions. |
| `data_cleaning.py` | Data-cleaning source. |
| `main.py`, `templates/`, `static/` | Flask application and interface assets. |
| `requirement5.py` | Positive/negative sentence-span analysis and Excel exports. |
| `web_scraping/` | Data-collection source. |
| `data/` | Review datasets and lexical resources. |

## Reproduction

From the repository root, create a Python environment and install the packages needed for the component you are running. For the visualisation script, the imported third-party packages are `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `wordcloud` and `nltk`. The web interface additionally requires `flask`; Excel exports require `openpyxl`.

To regenerate charts from the included analysed CSV:

```sh
python visualisations.py
```

To start the existing web interface:

```sh
python main.py
```

The app is configured for `http://127.0.0.1:5000`. Recomputing the NLP pipeline may require NLTK tokenisation resources appropriate to the installed NLTK version. A fresh end-to-end execution and dependency lockfile have not been verified for this portfolio copy.

## Evaluation and limitations

- The evaluation reference converts review ratings of 1-5 into negative labels and 6-10 into positive labels. These are **proxy labels**, not independently annotated sentiment ground truth.
- The evaluation code classifies a zero sentiment score as positive; this treatment of neutral text affects interpretation.
- Reviews are a collected sample and do not represent every passenger. Unequal review counts, language, context and historical coverage can influence comparisons.
- Lexicon scoring can miss negation, sarcasm and context. Correlation and ranking differences do not establish causation.
- An existing chart-label issue remains: the diverging rank-difference chart title says sentiment rank minus official rank, while the code calculates official rank minus sentiment rank. The three selected charts above do not use that diverging-chart title.

## Team and provenance

Team P4-4: Balandra Dyanel Lancea Estillore, Daryl Lai Weng Hong, Derrick Ho Zong Xian, Ang Yi Ting Shermaine, and Cheow Khai Xuan Jordan.

Module instructors: Zhang Zhengchen and Eric Lam. Original team repository: [DyanelLancea/Airline-Review-Sentiment-Analysis](https://github.com/DyanelLancea/Airline-Review-Sentiment-Analysis). This portfolio copy retains the original project history and credits.
