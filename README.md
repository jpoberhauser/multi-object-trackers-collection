# Multi-Object Tracking


- [Multi-Object Tracking](#multi-object-tracking)
  - [Surveys](#surveys)
    - [2022](#2022)
      - [Recent Advances in Embedding Methods for Multi-Object Tracking: A Survey](#recent-advances-in-embedding-methods-for-multi-object-tracking-a-survey)
    - [2021](#2021)
      - [Do Different Tracking Tasks Require Different Appearance Models?](#do-different-tracking-tasks-require-different-appearance-models)
    - [2019](#2019)
      - [Deep Learning in Video Multi-Object Tracking: A Survey](#deep-learning-in-video-multi-object-tracking-a-survey)
  - [Tracking by Detection](#tracking-by-detection)
    - [2024](#2024)
      - [Hybrid-SORT](#hybrid-sort)
      - [BoostTrack and BoosTrack++](#boosttrack-and-boostrack)
    - [2023](#2023)
      - [MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking](#memotr-long-term-memory-augmented-transformer-for-multi-object-tracking)
      - [GHOST](#ghost)
      - [OCSort](#ocsort)
      - [SUSHI](#sushi)
    - [2022](#2022-1)
      - [StrongSort](#strongsort)
      - [Bot-Sort](#bot-sort)
      - [QDTrack: Quasi-Dense Similarity Learning for Appearance-Only Multiple Object Tracking](#qdtrack-quasi-dense-similarity-learning-for-appearance-only-multiple-object-tracking)
      - [MeMOT: Multi-Object Tracking with Memory](#memot-multi-object-tracking-with-memory)
    - [2021](#2021-1)
      - [Online Multiple Object Tracking with Cross-Task Synergy](#online-multiple-object-tracking-with-cross-task-synergy)
      - [Learning a Proposal Classifier for Multiple Object Tracking](#learning-a-proposal-classifier-for-multiple-object-tracking)
      - [ByteTrack](#bytetrack)
      - [Tracking Objects as Points](#tracking-objects-as-points)
    - [2020](#2020)
      - [FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](#fairmot-on-the-fairness-of-detection-and-re-identification-in-multiple-object-tracking)
      - [A Unified Object Motion and Affinity Model for Online Multi-Object Tracking](#a-unified-object-motion-and-affinity-model-for-online-multi-object-tracking)
      - [Towards Real-Time Multi-Object Tracking](#towards-real-time-multi-object-tracking)
      - [Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking](#probabilistic-tracklet-scoring-and-inpainting-for-multiple-object-tracking)
    - [2019](#2019-1)
      - [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](#bag-of-tricks-and-a-strong-baseline-for-deep-person-re-identification)
    - [2017](#2017)
      - [DeepSort](#deepsort)
      - [Tracking The Untrackable: Learning To Track Multiple Cues with Long-Term Dependencies](#tracking-the-untrackable-learning-to-track-multiple-cues-with-long-term-dependencies)
    - [2016](#2016)
      - [Person Re-Identification by Multi-Channel Parts-Based CNN with Improved Triplet Loss Function](#person-re-identification-by-multi-channel-parts-based-cnn-with-improved-triplet-loss-function)
      - [Learning by tracking: Siamese CNN for robust target association](#learning-by-tracking-siamese-cnn-for-robust-target-association)
      - [SORT](#sort)
  - [Joint Tracking and ID](#joint-tracking-and-id)
    - [2024](#2024-1)
      - [Multiple Object Tracking as ID Prediction](#multiple-object-tracking-as-id-prediction)
    - [2022](#2022-2)
      - [Global Tracking Transformers](#global-tracking-transformers)
    - [2021](#2021-2)
      - [Joint Object Detection and Multi-Object Tracking with Graph Neural Networks](#joint-object-detection-and-multi-object-tracking-with-graph-neural-networks)
      - [Track to Detect and Segment: An Online Multi-Object Tracker](#track-to-detect-and-segment-an-online-multi-object-tracker)
    - [2020](#2020-1)
      - [TransTrack: Multiple Object Tracking with Transformer](#transtrack-multiple-object-tracking-with-transformer)
      - [Learning a Neural Solver for Multiple Object Tracking](#learning-a-neural-solver-for-multiple-object-tracking)
      - [RetinaTrack: Online Single Stage Joint Detection and Tracking](#retinatrack-online-single-stage-joint-detection-and-tracking)
      - [TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model](#tubetk-adopting-tubes-to-track-multi-object-in-a-one-step-training-model)
      - [Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking](#chained-tracker-chaining-paired-attentive-regression-results-for-end-to-end-joint-multiple-object-detection-and-tracking)
    - [2019](#2019-2)
      - [Tracking without bells and whistles](#tracking-without-bells-and-whistles)
  - [End-to-End](#end-to-end)
    - [2022](#2022-3)
      - [MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors](#motrv2-bootstrapping-end-to-end-multi-object-tracking-by-pretrained-object-detectors)
      - [TrackFormer](#trackformer)
    - [2021](#2021-3)
      - [MOTR: End-to-End Multiple-Object Tracking with Transformer](#motr-end-to-end-multiple-object-tracking-with-transformer)
      - [Learning to Track with Object Permanence](#learning-to-track-with-object-permanence)
    - [2020](#2020-2)
      - [Simple Unsupervised Multi-Object Tracking](#simple-unsupervised-multi-object-tracking)
    - [2019](#2019-3)
      - [How To Train Your Deep Multi-Object Tracker](#how-to-train-your-deep-multi-object-tracker)
  - [3D Tracking](#3d-tracking)
    - [2024](#2024-2)
      - [MCTrack: A Unified 3D Multi-Object Tracking Framework for Autonomous Driving](#mctrack-a-unified-3d-multi-object-tracking-framework-for-autonomous-driving)


This repo provides a list of research papers on Multi-Object Tracking (MOT). The post is organized by the publication year of each paper, starting from 2016 to 2024.
Within each year, the post lists out the title of each paper with its corresponding links to the paper on arxiv and the code implementation.


## Surveys

### 2022

#### Recent Advances in Embedding Methods for Multi-Object Tracking: A Survey

[paper](https://arxiv.org/abs/2205.10766)

### 2021

#### Do Different Tracking Tasks Require Different Appearance Models?

[paper](https://arxiv.org/abs/2107.02156)

[python code](https://github.com/Zhongdao/UniTrack)


### 2019

#### Deep Learning in Video Multi-Object Tracking: A Survey


[paper](https://arxiv.org/pdf/1907.12740)

## Tracking by Detection

### 2024

#### Hybrid-SORT

[paper](https://arxiv.org/abs/2308.00783)

[python code](https://github.com/ymzis69/HybridSORT)

#### BoostTrack and BoosTrack++

[paper](https://www.researchgate.net/publication/379780388_BoostTrack_boosting_the_similarity_measure_and_detection_confidence_for_improved_multiple_object_tracking)

[python code](https://github.com/vukasin-stanojevic/BoostTrack)

[paper](https://arxiv.org/abs/2408.13003)

### 2023

#### MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking

[paper](https://arxiv.org/abs/2307.15700)

[python code](https://github.com/MCG-NJU/MeMOTR)

#### GHOST

[paper](https://arxiv.org/abs/2206.04656)

[python code](https://github.com/dvl-tum/GHOST)

#### OCSort

[paper](https://arxiv.org/abs/2203.14360)

[python code](https://github.com/noahcao/OC_SORT)

#### SUSHI

[paper](https://arxiv.org/abs/2212.03038)

[python code](https://github.com/dvl-tum/SUSHI)

### 2022

#### StrongSort

[paper](https://arxiv.org/abs/2202.13514)

[python code](https://github.com/dyhBUPT/StrongSORT/tree/master)

#### Bot-Sort

[paper](https://arxiv.org/abs/2206.14651)

[python code -- boxmot package](https://github.com/mikel-brostrom/boxmot/tree/master)

#### QDTrack: Quasi-Dense Similarity Learning for Appearance-Only Multiple Object Tracking


[paper]https://arxiv.org/abs/2210.06984

[python code](https://github.com/SysCV/qdtrack)

#### MeMOT: Multi-Object Tracking with Memory


[paper](https://arxiv.org/abs/2203.16761)

### 2021

#### Online Multiple Object Tracking with Cross-Task Synergy


[paper](https://arxiv.org/abs/2104.00380)

[python code](https://github.com/songguocode/TADAM)

#### Learning a Proposal Classifier for Multiple Object Tracking


[paper](https://arxiv.org/abs/2103.07889)

[python code](https://github.com/daip13/LPC_MOT)

#### ByteTrack

[paper](https://arxiv.org/abs/2110.06864)

[python code -- boxmot package](https://github.com/mikel-brostrom/boxmot/tree/master)


#### Tracking Objects as Points

[paper](https://arxiv.org/abs/2004.01177)

[python code](https://github.com/xingyizhou/CenterTrack)


### 2020

#### FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking

[paper](https://arxiv.org/abs/2004.01888)

[python code](https://github.com/ifzhang/FairMOT)

#### A Unified Object Motion and Affinity Model for Online Multi-Object Tracking


[paper](https://arxiv.org/abs/2003.11291)

[python code](https://github.com/yinjunbo/UMA-MOT)

#### Towards Real-Time Multi-Object Tracking

[paper](https://arxiv.org/abs/1909.12605)

[python code](https://github.com/Zhongdao/Towards-Realtime-MOT)

#### Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking

[paper]https://arxiv.org/abs/2012.02337

### 2019

#### Bag of Tricks and A Strong Baseline for Deep Person Re-identification

[paper](https://arxiv.org/abs/1903.07071)

[python code](https://github.com/michuanhaohao/reid-strong-baseline)

### 2017

#### DeepSort

[paper](https://arxiv.org/abs/1703.07402)

[python code](https://github.com/nwojke/deep_sort)

#### Tracking The Untrackable: Learning To Track Multiple Cues with Long-Term Dependencies


[paper]https://arxiv.org/abs/1701.01909

### 2016

#### Person Re-Identification by Multi-Channel Parts-Based CNN with Improved Triplet Loss Function

[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Cheng_Person_Re-Identification_by_CVPR_2016_paper.pdf)

#### Learning by tracking: Siamese CNN for robust target association


[paper]https://arxiv.org/abs/1604.07866

#### SORT


[paper](https://arxiv.org/abs/1602.00763)

[python code](https://github.com/abewley/sort)




## Joint Tracking and ID

### 2024

#### Multiple Object Tracking as ID Prediction

[paper](https://arxiv.org/abs/2403.16848)

[python code](https://github.com/MCG-NJU/MOTIP)

### 2022

#### Global Tracking Transformers


[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Global_Tracking_Transformers_CVPR_2022_paper.pdf)

[python code](https://github.com/xingyizhou/GTR)

### 2021

#### Joint Object Detection and Multi-Object Tracking with Graph Neural Networks

[paper](https://arxiv.org/abs/2006.13164)

[python code](https://github.com/yongxinw/GSDT)

#### Track to Detect and Segment: An Online Multi-Object Tracker


[paper](https://arxiv.org/abs/2103.08808)

[python code](https://github.com/JialianW/TraDeS)

### 2020


#### TransTrack: Multiple Object Tracking with Transformer

[paper](https://arxiv.org/abs/2012.15460)

[python code](https://github.com/PeizeSun/TransTrack)

#### Learning a Neural Solver for Multiple Object Tracking

[paper]https://arxiv.org/abs/1912.07515

[python code](https://github.com/dvl-tum/mot_neural_solver)


#### RetinaTrack: Online Single Stage Joint Detection and Tracking

[paper](https://arxiv.org/abs/2003.13870)

[pytorch code](https://github.com/Hanson0910/RetinaTrack)

#### TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model


[paper]https://arxiv.org/abs/2006.05683

[python code](https://github.com/BoPang1996/TubeTK)

#### Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking

[paper](https://arxiv.org/abs/2007.14557)

[python code](https://github.com/pjl1995/CTracker)

### 2019

#### Tracking without bells and whistles

[paper](https://arxiv.org/pdf/1903.05625)

[python code](https://github.com/phil-bergmann/tracking_wo_bnw)





## End-to-End 



### 2022

#### MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors


[paper](https://arxiv.org/abs/2211.09791)

[python code](https://github.com/megvii-research/MOTRv2?tab=readme-ov-file)

#### TrackFormer

[paper](https://arxiv.org/abs/2101.02702)

[python code](https://github.com/timmeinhardt/trackformer.)


### 2021

#### MOTR: End-to-End Multiple-Object Tracking with Transformer

[paper](https://arxiv.org/abs/2105.03247)

[python code](https://github.com/megvii-research/MOTR)

#### Learning to Track with Object Permanence


[paper](https://arxiv.org/abs/2103.14258)


### 2020


#### Simple Unsupervised Multi-Object Tracking

[paper](https://arxiv.org/abs/2006.02609)

### 2019

#### How To Train Your Deep Multi-Object Tracker


[paper](https://arxiv.org/abs/1906.06618)

[python code](https://github.com/yihongXU/deepMOT)


## 3D Tracking

### 2024

#### MCTrack: A Unified 3D Multi-Object Tracking Framework for Autonomous Driving


[paper](https://arxiv.org/abs/2409.16149)

[python code](https://github.com/megvii-research/MCTrack)