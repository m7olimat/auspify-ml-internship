# Automated insights

- The catalogue holds 8,790 titles: 70% movies and 30% TV shows.
- Most common genres: International Movies (2,752), Dramas (2,426), Comedies (1,674).
- Biggest producing countries: United States (3,240), India (1,057), United Kingdom (638).
- 55% of titles are 'fresh releases' (added within a year of release). TV shows are more often fresh (67%) than movies (50%).
- Genres with the highest fresh share: TV Action & Adventure (80%, n=167), Stand-Up Comedy (78%, n=343), Crime TV Shows (77%, n=469).
- Genres leaning on older titles: Action & Adventure (34%, n=859), Sci-Fi & Fantasy (36%, n=243), Romantic Movies (40%, n=616).
- Countries (100+ titles) with the freshest content: Spain (89%, n=182), China (77%, n=100); the oldest-leaning: Egypt (17%, n=123), Turkey (25%, n=112).
- By audience: adult 63%, kids 59%, family/teen 45% fresh.
- Fresh share moved from 66% in 2016 to 51% in 2021 (years with 100+ additions).
- Best model: Gradient Boosting (test ROC-AUC 0.81, accuracy 74%, F1 0.78).
- Attributes that matter most to the model: rating, main country, year added, duration min, month added.
- Reminder: 'fresh release' is a proxy, not real viewership. Treat these as patterns in how the catalogue is built, not as proof of what audiences like.
