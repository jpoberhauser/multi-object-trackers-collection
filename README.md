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
  - [Animal Tracking / Implementation](#animal-tracking--implementation)
      - [An HMM-based framework for identity-aware long-term multi-object tracking from sparse and uncertain identification: use case on long-term tracking in livestock](#an-hmm-based-framework-for-identity-aware-long-term-multi-object-tracking-from-sparse-and-uncertain-identification-use-case-on-long-term-tracking-in-livestock)


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

For a long time, the most common paradigm in Multi-Object Tracking was tracking-by-detection (TbD), where objects are first detected and then associated over video frames. For association, most models resourced to motion and appearance cues, e.g., re-identification networks. Recent approaches based on attention propose to learn the cues in a data-driven manner, showing impressive results. In this paper, we ask ourselves whether simple good old TbD methods are also capable of achieving the performance of end-to-end models. To this end, we propose two key ingredients that allow a standard re-identification network to excel at appearance-based tracking. We extensively analyse its failure cases, and show that a combination of our appearance features with a simple motion model leads to strong tracking results. Our tracker generalizes to four public datasets, namely MOT17, MOT20, BDD100k, and DanceTrack, achieving state-of-the-art performance

[paper](https://arxiv.org/abs/2206.04656)

[python code](https://github.com/dvl-tum/GHOST)

#### OCSort

Kalman filter (KF) based methods for multi-object tracking (MOT) make an assumption that objects move linearly. While this assumption is acceptable for very short periods of occlusion, linear estimates of motion for prolonged time can be highly inaccurate. Moreover, when there is no measurement available to update Kalman filter parameters, the standard convention is to trust the priori state estimations for posteriori update. This leads to the accumulation of errors during a period of occlusion. The error causes significant motion direction variance in practice. In this work, we show that a basic Kalman filter can still obtain state-of-the-art tracking performance if proper care is taken to fix the noise accumulated during occlusion. Instead of relying only on the linear state estimate (i.e., estimation-centric approach), we use object observations (i.e., the measurements by object detector) to compute a virtual trajectory over the occlusion period to fix the error accumulation of filter parameters during the occlusion period. This allows more time steps to correct errors accumulated during occlusion. We name our method Observation-Centric SORT (OC-SORT). It remains Simple, Online, and Real-Time but improves robustness during occlusion and non-linear motion. Given off-the-shelf detections as input, OC-SORT runs at 700+ FPS on a single CPU. It achieves state-of-the-art on multiple datasets, including MOT17, MOT20, KITTI, head tracking, and especially DanceTrack where the object motion is highly non-linear

[paper](https://arxiv.org/abs/2203.14360)

[python code](https://github.com/noahcao/OC_SORT)

#### SUSHI

Tracking objects over long videos effectively means solving a spectrum of problems, from short-term association for un-occluded objects to long-term association for objects that are occluded and then reappear in the scene. Methods tackling these two tasks are often disjoint and crafted for specific scenarios, and top-performing approaches are often a mix of techniques, which yields engineering-heavy solutions that lack generality. In this work, we question the need for hybrid approaches and introduce SUSHI, a unified and scalable multi-object tracker. Our approach processes long clips by splitting them into a hierarchy of subclips, which enables high scalability. We leverage graph neural networks to process all levels of the hierarchy, which makes our model unified across temporal scales and highly general. As a result, we obtain significant improvements over state-of-the-art on four diverse datasets

[paper](https://arxiv.org/abs/2212.03038)

[python code](https://github.com/dvl-tum/SUSHI)

### 2022

#### StrongSort

Recently, Multi-Object Tracking (MOT) has attracted rising attention, and accordingly, remarkable progresses have been achieved. However, the existing methods tend to use various basic models (e.g, detector and embedding model), and different training or inference tricks, etc. As a result, the construction of a good baseline for a fair comparison is essential. In this paper, a classic tracker, i.e., DeepSORT, is first revisited, and then is significantly improved from multiple perspectives such as object detection, feature embedding, and trajectory association. The proposed tracker, named StrongSORT, contributes a strong and fair baseline for the MOT community. Moreover, two lightweight and plug-and-play algorithms are proposed to address two inherent "missing" problems of MOT: missing association and missing detection. Specifically, unlike most methods, which associate short tracklets into complete trajectories at high computation complexity, we propose an appearance-free link model (AFLink) to perform global association without appearance information, and achieve a good balance between speed and accuracy. Furthermore, we propose a Gaussian-smoothed interpolation (GSI) based on Gaussian process regression to relieve the missing detection. AFLink and GSI can be easily plugged into various trackers with a negligible extra computational cost (1.7 ms and 7.1 ms per image, respectively, on MOT17). Finally, by fusing StrongSORT with AFLink and GSI, the final tracker (StrongSORT++) achieves state-of-the-art results on multiple public benchmarks, i.e., MOT17, MOT20, DanceTrack and KITTI.

[paper](https://arxiv.org/abs/2202.13514)

[python code](https://github.com/dyhBUPT/StrongSORT/tree/master)

#### Bot-Sort

The goal of multi-object tracking (MOT) is detecting and tracking all the objects in a scene, while keeping a unique identifier for each object. In this paper, we present a new robust state-of-the-art tracker, which can combine the advantages of motion and appearance information, along with camera-motion compensation, and a more accurate Kalman filter state vector. Our new trackers BoT-SORT, and BoT-SORT-ReID rank first in the datasets of MOTChallenge [29, 11] on both MOT17 and MOT20 test sets, in terms of all the main MOT metrics: MOTA, IDF1, and HOTA. For MOT17: 80.5 MOTA, 80.2 IDF1, and 65.0 HOTA are achieved.

[paper](https://arxiv.org/abs/2206.14651)

[python code -- boxmot package](https://github.com/mikel-brostrom/boxmot/tree/master)

#### QDTrack: Quasi-Dense Similarity Learning for Appearance-Only Multiple Object Tracking

Similarity learning has been recognized as a crucial step for object tracking. However, existing multiple object tracking methods only use sparse ground truth matching as the training objective, while ignoring the majority of the informative regions in images. In this paper, we present Quasi-Dense Similarity Learning, which densely samples hundreds of object regions on a pair of images for contrastive learning. We combine this similarity learning with multiple existing object detectors to build Quasi-Dense Tracking (QDTrack), which does not require displacement regression or motion priors. We find that the resulting distinctive feature space admits a simple nearest neighbor search at inference time for object association. In addition, we show that our similarity learning scheme is not limited to video data, but can learn effective instance similarity even from static input, enabling a competitive tracking performance without training on videos or using tracking supervision. We conduct extensive experiments on a wide variety of popular MOT benchmarks. We find that, despite its simplicity, QDTrack rivals the performance of state-of-the-art tracking methods on all benchmarks and sets a new state-of-the-art on the large-scale BDD100K MOT benchmark, while introducing negligible computational overhead to the detector.

[paper](https://arxiv.org/abs/2210.06984)

[python code](https://github.com/SysCV/qdtrack)

#### MeMOT: Multi-Object Tracking with Memory

We propose an online tracking algorithm that performs the object detection and data association under a common framework, capable of linking objects after a long time span. This is realized by preserving a large spatio-temporal memory to store the identity embeddings of the tracked objects, and by adaptively referencing and aggregating useful information from the memory as needed. Our model, called MeMOT, consists of three main modules that are all Transformer-based: 1) Hypothesis Generation that produce object proposals in the current video frame; 2) Memory Encoding that extracts the core information from the memory for each tracked object; and 3) Memory Decoding that solves the object detection and data association tasks simultaneously for multi-object tracking. When evaluated on widely adopted MOT benchmark datasets, MeMOT observes very competitive performance.

[paper](https://arxiv.org/abs/2203.16761)

### 2021

#### Online Multiple Object Tracking with Cross-Task Synergy

Modern online multiple object tracking (MOT) methods usually focus on two directions to improve tracking performance. One is to predict new positions in an incoming frame based on tracking information from previous frames, and the other is to enhance data association by generating more discriminative identity embeddings. Some works combined both directions within one framework but handled them as two individual tasks, thus gaining little mutual benefits. In this paper, we propose a novel unified model with synergy between position prediction and embedding association. The two tasks are linked by temporal-aware target attention and distractor attention, as well as identity-aware memory aggregation model. Specifically, the attention modules can make the prediction focus more on targets and less on distractors, therefore more reliable embeddings can be extracted accordingly for association. On the other hand, such reliable embeddings can boost identity-awareness through memory aggregation, hence strengthen attention modules and suppress drifts. In this way, the synergy between position prediction and embedding association is achieved, which leads to strong robustness to occlusions. Extensive experiments demonstrate the superiority of our proposed model over a wide range of existing methods on MOTChallenge benchmarks.

[paper](https://arxiv.org/abs/2104.00380)

[python code](https://github.com/songguocode/TADAM)

#### Learning a Proposal Classifier for Multiple Object Tracking

The recent trend in multiple object tracking (MOT) is heading towards leveraging deep learning to boost the tracking performance. However, it is not trivial to solve the data-association problem in an end-to-end fashion. In this paper, we propose a novel proposal-based learnable framework, which models MOT as a proposal generation, proposal scoring and trajectory inference paradigm on an affinity graph. This framework is similar to the two-stage object detector Faster RCNN, and can solve the MOT problem in a data-driven way. For proposal generation, we propose an iterative graph clustering method to reduce the computational cost while maintaining the quality of the generated proposals. For proposal scoring, we deploy a trainable graph-convolutional-network (GCN) to learn the structural patterns of the generated proposals and rank them according to the estimated quality scores. For trajectory inference, a simple deoverlapping strategy is adopted to generate tracking output while complying with the constraints that no detection can be assigned to more than one track. We experimentally demonstrate that the proposed method achieves a clear performance improvement in both MOTA and IDF1 with respect to previous state-of-the-art on two public benchmarks.

[paper](https://arxiv.org/abs/2103.07889)

[python code](https://github.com/daip13/LPC_MOT)

#### ByteTrack

Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, we present a simple, effective and generic association method, tracking by associating almost every detection box instead of only the high score ones. For the low score detection boxes, we utilize their similarities with tracklets to recover true objects and filter out the background detections. When applied to 9 different state-of-the-art trackers, our method achieves consistent improvement on IDF1 score ranging from 1 to 10 points. To put forwards the state-of-the-art performance of MOT, we design a simple and strong tracker, named ByteTrack. For the first time, we achieve 80.3 MOTA, 77.3 IDF1 and 63.1 HOTA on the test set of MOT17 with 30 FPS running speed on a single V100 GPU. ByteTrack also achieves state-of-the-art performance on MOT20, HiEve and BDD100K tracking benchmarks. 

[paper](https://arxiv.org/abs/2110.06864)

[python code -- boxmot package](https://github.com/mikel-brostrom/boxmot/tree/master)


#### Tracking Objects as Points

Tracking has traditionally been the art of following interest points through space and time. This changed with the rise of powerful deep networks. Nowadays, tracking is dominated by pipelines that perform object detection followed by temporal association, also known as tracking-by-detection. In this paper, we present a simultaneous detection and tracking algorithm that is simpler, faster, and more accurate than the state of the art. Our tracker, CenterTrack, applies a detection model to a pair of images and detections from the prior frame. Given this minimal input, CenterTrack localizes objects and predicts their associations with the previous frame. That's it. CenterTrack is simple, online (no peeking into the future), and real-time. It achieves 67.3% MOTA on the MOT17 challenge at 22 FPS and 89.4% MOTA on the KITTI tracking benchmark at 15 FPS, setting a new state of the art on both datasets. CenterTrack is easily extended to monocular 3D tracking by regressing additional 3D attributes. Using monocular video input, it achieves 28.3% AMOTA@0.2 on the newly released nuScenes 3D tracking benchmark, substantially outperforming the monocular baseline on this benchmark while running at 28 FPS

[paper](https://arxiv.org/abs/2004.01177)

[python code](https://github.com/xingyizhou/CenterTrack)


### 2020

#### FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking

Multi-object tracking (MOT) is an important problem in computer vision which has a wide range of applications. Formulating MOT as multi-task learning of object detection and re-ID in a single network is appealing since it allows joint optimization of the two tasks and enjoys high computation efficiency. However, we find that the two tasks tend to compete with each other which need to be carefully addressed. In particular, previous works usually treat re-ID as a secondary task whose accuracy is heavily affected by the primary detection task. As a result, the network is biased to the primary detection task which is not fair to the re-ID task. To solve the problem, we present a simple yet effective approach termed as FairMOT based on the anchor-free object detection architecture CenterNet. Note that it is not a naive combination of CenterNet and re-ID. Instead, we present a bunch of detailed designs which are critical to achieve good tracking results by thorough empirical studies. The resulting approach achieves high accuracy for both detection and tracking. The approach outperforms the state-of-the-art methods by a large margin on several public datasets.

[paper](https://arxiv.org/abs/2004.01888)

[python code](https://github.com/ifzhang/FairMOT)

#### A Unified Object Motion and Affinity Model for Online Multi-Object Tracking

Current popular online multi-object tracking (MOT) solutions apply single object trackers (SOTs) to capture object motions, while often requiring an extra affinity network to associate objects, especially for the occluded ones. This brings extra computational overhead due to repetitive feature extraction for SOT and affinity computation. Meanwhile, the model size of the sophisticated affinity network is usually non-trivial. In this paper, we propose a novel MOT framework that unifies object motion and affinity model into a single network, named UMA, in order to learn a compact feature that is discriminative for both object motion and affinity measure. In particular, UMA integrates single object tracking and metric learning into a unified triplet network by means of multi-task learning. Such design brings advantages of improved computation efficiency, low memory requirement and simplified training procedure. In addition, we equip our model with a task-specific attention module, which is used to boost task-aware feature learning. The proposed UMA can be easily trained end-to-end, and is elegant - requiring only one training stage. Experimental results show that it achieves promising performance on several MOT Challenge benchmarks.

[paper](https://arxiv.org/abs/2003.11291)

[python code](https://github.com/yinjunbo/UMA-MOT)

#### Towards Real-Time Multi-Object Tracking

Modern multiple object tracking (MOT) systems usually follow the \emph{tracking-by-detection} paradigm. It has 1) a detection model for target localization and 2) an appearance embedding model for data association. Having the two models separately executed might lead to efficiency problems, as the running time is simply a sum of the two steps without investigating potential structures that can be shared between them. Existing research efforts on real-time MOT usually focus on the association step, so they are essentially real-time association methods but not real-time MOT system. In this paper, we propose an MOT system that allows target detection and appearance embedding to be learned in a shared model. Specifically, we incorporate the appearance embedding model into a single-shot detector, such that the model can simultaneously output detections and the corresponding embeddings. We further propose a simple and fast association method that works in conjunction with the joint model. In both components the computation cost is significantly reduced compared with former MOT systems, resulting in a neat and fast baseline for future follow-ups on real-time MOT algorithm design. To our knowledge, this work reports the first (near) real-time MOT system, with a running speed of 22 to 40 FPS depending on the input resolution. Meanwhile, its tracking accuracy is comparable to the state-of-the-art trackers embodying separate detection and embedding (SDE) learning (64.4% MOTA \vs 66.1% MOTA on MOT-16 challenge)

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


## Animal Tracking / Implementation

#### An HMM-based framework for identity-aware long-term multi-object tracking from sparse and uncertain identification: use case on long-term tracking in livestock

[paper](https://drive.google.com/file/d/1_-6oLD4X2FHp3bo-Qp4PDtcpEMWr0kIL/view)

[python code](https://github.com/ngobibibnbe/uncertain-identity-aware-tracking)