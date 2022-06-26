University of Zagreb, Faculty of Electrical Engineering and Computing

## Stress Detection in Social Media

##### Fani Sentinella-Jerbić, Dorian Smoljan, Vladimir Rzaev, 2022

Social media users are active more than ever, posting new content on daily. On the other hand, stress has become extremely prominent and easily observable at the same time. Given the data collected on Reddit and related work done on it, we built a system that would recognize posts riddled with stress markers, i.e., states of mental or emotional tension. Essentially, the system boils down to classifying whether a post displays aspects of stress or not.

Specifically, we improved upon system presented by [Knežević et al.](https://www.fer.unizg.hr/_download/repository/TAR-2021-ProjectReports.pdf#page=48)
We managed to improve it through three modifications:
- Using unlabeled data to adapt RoBERTa to the task domain,
- Replacing LIWC features with Empath features,
- Experimenting with different ML algorithms and hyperparameter settings.



#### Our final model architecture:

![System](https://github.com/fsentin/dreaddit/blob/main/system.png)
