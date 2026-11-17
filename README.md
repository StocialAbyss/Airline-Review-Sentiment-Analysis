# Airline Review Sentiment Analysis

**Python | Pandas | Matplotlib | NLP | Flask**

An academic team project for **INF1002: Programming Fundamentals** at the Singapore Institute of Technology, Team P4-4.

The project turns airline review text into sentiment scores, compares patterns across airlines and over time, and presents selected reviews through a Flask web interface. The workflow covers data collection, cleaning, text analysis, evaluation and visual explanation.

## What question does the project answer?

How do passengers describe their airline experiences, and what patterns become visible when those reviews are analysed together?

The project explores how average sentiment differs between airlines, how much reviews vary within each airline, how sentiment changes over time, and how a review-derived ranking compares with the external Skytrax ranking recorded in the project.

## 1. Collect and prepare the reviews

The data-collection and cleaning code prepares review text, airline names, ratings and publication dates for analysis. The repository includes raw, cleaned and analysed CSV files.

Preparing consistent fields makes it possible to group reviews by airline, organise results by month and compare text-derived sentiment with ratings. A separate dynamic-programming component supports word segmentation for text containing missing spaces.

## 2. Calculate sentiment and identify review passages

The sentiment pipeline uses the **AFINN lexicon**, which maps recognised words to positive or negative scores.

1. Split review text into words and look up their sentiment scores.
2. Sum the word scores to produce a raw review score.
3. Divide by the word count and clamp the result to the range -1 to 1.
4. Identify strongly positive and negative sentences and fixed-length word windows.
5. Export the analysed reviews for visualisation and the web interface.

An additional component uses Kadane's algorithm on sentence scores to find contiguous positive and negative sentence spans and export them to Excel.

This is **lexicon-based sentiment analysis**: the application does not train a new language model or use generative AI to write its findings.

## 3. Compare airlines through graphs

### External ranking versus sentiment-derived ranking

![Grouped horizontal bars comparing external airline ranks with sentiment-derived ranks](visualisations/Official_vs_Sentiment.png)

Average normalised sentiment is calculated for each airline and converted into a ranking. The grouped bars then compare that ranking with the external list recorded in the project. **A smaller rank number is better.**

The chart makes differences easy to identify. For example, Hainan Airlines appears tenth in the external list but first in the review-derived ranking. This highlights disagreement between two different measures; it does not establish that one ranking is correct or that the collected reviews represent every passenger.

### Distribution of sentiment within each airline

![Boxplots of normalised review sentiment by airline, with median lines and green mean markers](visualisations/boxplot_sentiment_consistency.png)

Averages can hide substantial variation between reviews. These boxplots show the median, the middle 50% of scores and statistical outliers. Green triangles mark the means.

**What to look for:** differences in the centre, box width and tails. An airline can have a positive average while still receiving a wide range of reviews. Points beyond the whiskers are unusual relative to the distribution, rather than automatically being data errors.

### Changes in monthly sentiment

![Monthly sentiment trends with Singapore Airlines highlighted against other airlines](visualisations/trend_spotlight.png)

Reviews are grouped into monthly average scores, then smoothed with a rolling window. Singapore Airlines is highlighted with a thicker line so its movement is easier to follow among the comparison airlines.

The chart is useful for exploring changes that deserve further investigation. Review volumes, missing months and changes in the sample should be checked before attributing a rise or fall to a particular event. The implementation uses three rows of monthly observations; a strict three-calendar-month interpretation requires checking for gaps.

### Relationship between external rank and average sentiment

![Scatterplot of external Skytrax rank against average review sentiment for ten airlines](visualisations/scatter_official_rank_vs_sentiment.png)

Each point represents an airline. The vertical axis shows average normalised sentiment and the horizontal axis shows external rank; **rank 1 is on the right** in this saved chart.

The displayed Pearson correlation is approximately **0.18**, indicating a weak linear relationship in this ten-airline comparison. The chart supports examining agreement between the measures, rather than assuming that a highly ranked airline must also have the highest sentiment in this sample.

### Additional text summaries

The visualisation script also exports positive/negative word clouds and TF-IDF keyword summaries. These help explore the vocabulary used in reviews, but frequent terms alone do not establish the cause of a customer's experience.

## 4. Evaluate sentiment classification

The evaluation converts scores into positive/negative predictions and compares them with rating-derived labels:

- Ratings **1-5** form the negative reference class.
- Ratings **6-10** form the positive reference class.
- Sentiment scores below zero are negative; scores of zero or above are positive.

The [saved evaluation CSV](visualisations/Sentiment_Analysis_Metrics.csv) reports:

| Metric | Saved result | What it describes |
| --- | --- | --- |
| Accuracy | 80.45% | Overall agreement with rating-derived labels. |
| Precision | 77.72% | How often a positive prediction matches a positive reference label. |
| Recall | 93.14% | How many positive reference labels are identified. |
| F1 score | 84.73% | The balance between positive-class precision and recall. |

These are saved project results, not a newly executed benchmark. Ratings are **proxy labels**, rather than independently annotated sentiment ground truth. Treating zero sentiment as positive also affects the evaluation.

## 5. Explore reviews in the web interface

The Flask application loads the analysed dataset and offers airline and month filters. It displays selected positive and negative reviews and lists leading airlines by average sentiment.

The month filter groups by month name across the dataset; it is not a separate year-and-month selector. The interface lets readers connect an aggregate result with individual review examples.

## Run the project

From the repository root, install the packages needed for the relevant component. The plotting script imports `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `wordcloud` and `nltk`. The web application also requires `flask`; Excel exports require `openpyxl`.

Regenerate the visualisations from the included analysed CSV:

```sh
python visualisations.py
```

Launch the local web interface:

```sh
python main.py
```

Open `http://127.0.0.1:5000`. Recomputing the NLP pipeline may require NLTK tokenisation resources appropriate to the installed version. These are source-based setup instructions; a fresh end-to-end installation has not been verified.

## Repository guide

| Location | Contents |
| --- | --- |
| `web_scraping/` | Review collection code. |
| `data_cleaning.py` | Data-cleaning code. |
| `sentiment_analysis.py` | Text preparation, scoring and passage analysis. |
| `requirement5.py` | Contiguous sentence-span analysis and Excel exports. |
| `visualisations.py` | Charts, keyword summaries and evaluation calculations. |
| `visualisations/` | Saved graphs and tabular outputs. |
| `main.py`, `templates/`, `static/` | Flask interface and assets. |
| `data/` | Review datasets and lexical resources. |

## Interpretation

The review sample can be affected by unequal volumes, historical coverage and selection bias. Dictionary scoring can miss context, negation and sarcasm. Ranking differences and correlations describe this dataset; they do not establish causal effects or definitive airline quality.

## Team and source

**Team P4-4:** Balandra Dyanel Lancea Estillore, Daryl Lai Weng Hong, Derrick Ho Zong Xian, Ang Yi Ting Shermaine, and Cheow Khai Xuan Jordan.

**Module instructors:** Zhang Zhengchen and Eric Lam.

Original repository: [DyanelLancea/Airline-Review-Sentiment-Analysis](https://github.com/DyanelLancea/Airline-Review-Sentiment-Analysis).
