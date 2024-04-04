## 第1个文件 similarity_ranking.json

这里 similarity_ranking.json 列出了当下任务(e.g, "smoker-status") 和一系列任务 (一共12个) 之间相似度的排名, 这个列表可能是基于某种相似性度量（例如，基于任务描述、数据特征或已知解决方案的相似性）来排名的:
```
{
    "smoker-status": [
        "enzyme-substrate",
        "spaceship-titanic",
        "airline-reviews",
        "ethanol-concentration",
        "chatgpt-prompt",
...
    ],
    "mohs-hardness": [
        "media-campaign-cost",
        "wild-blueberry-yield",
        "airline-reviews",
        "spaceship-titanic",
        "feedback",
...
    ],
```

这一系列任务都可以在 ./experience_replay 文件夹下面的其对应的代码, 例如：
```
-- deployment/
    --- config/
    --- experience_replay/
        |---- enzyme-substrate.py
        |---- spaceship-titanic.py
        |---- airline-reviews.py
        |---- ethanol-concentration.py
        |---- chatgpt-prompt.py
        |---- handwriting.py
        |---- wild-blueberry-yield.py
        |---- media-campaign-cost.py
        |---- textual-entailment.py
        |---- feedback.py
        |---- ett-m2.py
        |---- ili.py
-- development/
-- figures/

./experience_replay/enzyme-substrate.py,
```

## 第2个文件 heterogenous_similarity_ranking.json

heterogenous_similarity_ranking.json 则包含了更详细的信息，是 kaggle 比赛冠军🏆报告的一些解决方案总结。作者分享了他们团队的一些技巧和策略,比如数据预处理的方法、使用了哪些模型、集成方式等,并列出了单模型和集成模型在公开和私有数据集上的表现分数。最后还提到了一些他们没来得及尝试的想法。

* i.e, 它提供了一些特定任务 (一共18个) 的描述、所使用的模型、数据处理技巧、验证策略、损失函数、优化技巧以及其他一些可能有助于提高模型性能的技巧和想法。

```
{
    "smoker-status": "Thanks to the organizer and the Kaggle team for hosting this competition. And thanks to many participants who shared their ideas with notebook or discussion. It's difficult to improve the score until we find the \"magic\".

Fortunately, our team make the breakthrough and get 3rd place at the end of the competition. Great thanks to my teammates and their hard work! @xiamaozi11 @renxingkai @decalogue \n\n## Summary\n\nOur team tried to find the additional information about anchor and target in the [public dataset]( shared by the organizer.

However, this method has a little benefit because only part of them are matched or those texts are useless.\n\nThe essential part of our solution is adding targets with the same anchor to each data sample.

This data processing trick boosts our score from 0.84x to 0.85x on LB by a single model.\n\nWe stack 12 different models in the final submission. DeBERTa V3 large with MSE loss gives the best single model score on both CV and LB.

\n\n\n## Validation strategy\n\nBoth `StratifiedGroupKFold` and  `GroupKFold` can prevent data with the same anchor from leaking to validation set. `GroupKFold` can keep the same training data size of each fold, while `StratifiedGroupKFold` can keep the distribution of the label. Both of them are used (by different team member) and get relatively strong correlation between CV and LB.

\n\n\n## Data processing\n\nInput data from baseline\n```\nanchor [SEP] target [SEP] context text\n```\n\nOur input data\n```\nanchor [SEP] target; target_x1; target_x2; ... traget_xn; [SEP] context text\n```\nwhere target_xi are targets with the same anchor and context code.\n\nIt's easy to get comaprable improvement by hard encoding them while shuffling the sequence can reach higher score.\n\n\n## Model\n\n

Pretrained model\n- Electra large\n- Bert For Patent\n- DeBERTa V3 large\n- DeBERTa V1\n- DeBERTa V1 xlarge\n\nLoss\n- binary cross entropy loss\n- mean squared error loss\n- pearson correlation loss\n\nThere is no big difference among those loss functions.

However, using different loss in training phrases will lead to high diversity when ensembling because the distribution of the prediction looks different from oof.

\n\nTricks\n- different learning rate for different layer\n- fgm\n- ema\n\nYou may get around 1k~2k improvement by adding all of those tricks.\n\n## Result\n\nSingle Model\n\n| Model             | CV     | Public Score | ...
```


这些数据可能是为了帮助自动化数据科学平台（如DS-Agent）更好地理解不同任务之间的关系，并利用这些关系来提高模型的性能或简化模型开发过程。

通过分析这些相似性排名，DS-Agent可以在面对新任务时，更有效地检索和应用过去成功任务的知识和经验 => 来 掌握机器学习的一些技巧和trick。
