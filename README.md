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

Multi-object tracking (MOT) aims to associate target objects across video frames in order to obtain entire moving trajectories. With the advancement of deep neural networks and the increasing demand for intelligent video analysis, MOT has gained significantly increased interest in the computer vision community. Embedding methods play an essential role in object location estimation and temporal identity association in MOT. Unlike other computer vision tasks, such as image classification, object detection, re-identification, and segmentation, embedding methods in MOT have large variations, and they have never been systematically analyzed and summarized. In this survey, we first conduct a comprehensive overview with in-depth analysis for embedding methods in MOT from seven different perspectives, including patch-level embedding, single-frame embedding, cross-frame joint embedding, correlation embedding, sequential embedding, tracklet embedding, and cross-track relational embedding. We further summarize the existing widely used MOT datasets and analyze the advantages of existing state-of-the-art methods according to their embedding strategies. Finally, some critical yet under-investigated areas and future research directions are discussed.

[paper](https://arxiv.org/abs/2205.10766)

### 2021

#### Do Different Tracking Tasks Require Different Appearance Models?

Tracking objects of interest in a video is one of the most popular and widely applicable problems in computer vision. However, with the years, a Cambrian explosion of use cases and benchmarks has fragmented the problem in a multitude of different experimental setups. As a consequence, the literature has fragmented too, and now novel approaches proposed by the community are usually specialised to fit only one specific setup. To understand to what extent this specialisation is necessary, in this work we present UniTrack, a solution to address five different tasks within the same framework. UniTrack consists of a single and task-agnostic appearance model, which can be learned in a supervised or self-supervised fashion, and multiple ``heads'' that address individual tasks and do not require training. We show how most tracking tasks can be solved within this framework, and that the same appearance model can be successfully used to obtain results that are competitive against specialised methods for most of the tasks considered. The framework also allows us to analyse appearance models obtained with the most recent self-supervised methods, thus extending their evaluation and comparison to a larger variety of important problems.

[paper](https://arxiv.org/abs/2107.02156)

[python code](https://github.com/Zhongdao/UniTrack)


### 2019

#### Deep Learning in Video Multi-Object Tracking: A Survey

The problem of Multiple Object Tracking (MOT) consists in following the trajectory of different objects in a sequence, usually a video. In recent years, with the rise of Deep Learning, the algorithms that provide a solution to this problem have benefited from the representational power of deep models. This paper provides a comprehensive survey on works that employ Deep Learning models to solve
the task of MOT on single-camera videos. Four main steps in MOT algorithms are identified, and an in-depth review of how Deep Learning was employed in each one of these stages is presented. A complete experimental comparison of the presented works on the three MOTChallenge datasets is
also provided, identifying a number of similarities among the top-performing methods and presenting some possible future research directions.

[paper](https://arxiv.org/pdf/1907.12740)

## Tracking by Detection

### 2024

#### Hybrid-SORT

Multi-Object Tracking (MOT) aims to detect and associate all desired objects across frames. Most methods accomplish the task by explicitly or implicitly leveraging strong cues (i.e., spatial and appearance information), which exhibit powerful instance-level discrimination. However, when object occlusion and clustering occur, spatial and appearance information will become ambiguous simultaneously due to the high overlap among objects. In this paper, we demonstrate this long-standing challenge in MOT can be efficiently and effectively resolved by incorporating weak cues to compensate for strong cues. Along with velocity direction, we introduce the confidence and height state as potential weak cues. With superior performance, our method still maintains Simple, Online and Real-Time (SORT) characteristics. Also, our method shows strong generalization for diverse trackers and scenarios in a plug-and-play and training-free manner. Significant and consistent improvements are observed when applying our method to 5 different representative trackers. Further, with both strong and weak cues, our method Hybrid-SORT achieves superior performance on diverse benchmarks, including MOT17, MOT20, and especially DanceTrack where interaction and severe occlusion frequently happen with complex motions.

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

Despite the recent advances in multiple object tracking (MOT), achieved by joint detection and tracking, dealing with long occlusions remains a challenge. This is due to the fact that such techniques tend to ignore the long-term motion information. In this paper, we introduce a probabilistic autoregressive motion model to score tracklet proposals by directly measuring their likelihood. This is achieved by training our model to learn the underlying distribution of natural tracklets. As such, our model allows us not only to assign new detections to existing tracklets, but also to inpaint a tracklet when an object has been lost for a long time, e.g., due to occlusion, by sampling tracklets so as to fill the gap caused by misdetections. Our experiments demonstrate the superiority of our approach at tracking objects in challenging sequences; it outperforms the state of the art in most standard MOT metrics on multiple MOT benchmark datasets, including MOT16, MOT17, and MOT20

[paper](https://arxiv.org/abs/2012.02337)

### 2019

#### Bag of Tricks and A Strong Baseline for Deep Person Re-identification

This paper explores a simple and efficient baseline for person re-identification (ReID). Person re-identification (ReID) with deep neural networks has made progress and achieved high performance in recent years. However, many state-of-the-arts methods design complex network structure and concatenate multi-branch features. In the literature, some effective training tricks are briefly appeared in several papers or source codes. This paper will collect and evaluate these effective training tricks in person ReID. By combining these tricks together, the model achieves 94.5% rank-1 and 85.9% mAP on Market1501 with only using global features.

[paper](https://arxiv.org/abs/1903.07071)

[python code](https://github.com/michuanhaohao/reid-strong-baseline)

### 2017

#### DeepSort

Simple Online and Realtime Tracking (SORT) is a pragmatic approach to multiple object tracking with a focus on simple, effective algorithms. In this paper, we integrate appearance information to improve the performance of SORT. Due to this extension we are able to track objects through longer periods of occlusions, effectively reducing the number of identity switches. In spirit of the original framework we place much of the computational complexity into an offline pre-training stage where we learn a deep association metric on a large-scale person re-identification dataset. During online application, we establish measurement-to-track associations using nearest neighbor queries in visual appearance space. Experimental evaluation shows that our extensions reduce the number of identity switches by 45%, achieving overall competitive performance at high frame rates.

[paper](https://arxiv.org/abs/1703.07402)

[python code](https://github.com/nwojke/deep_sort)

#### Tracking The Untrackable: Learning To Track Multiple Cues with Long-Term Dependencies

The majority of existing solutions to the Multi-Target Tracking (MTT) problem do not combine cues in a coherent end-to-end fashion over a long period of time. However, we present an online method that encodes long-term temporal dependencies across multiple cues. One key challenge of tracking methods is to accurately track occluded targets or those which share similar appearance properties with surrounding objects. To address this challenge, we present a structure of Recurrent Neural Networks (RNN) that jointly reasons on multiple cues over a temporal window. We are able to correct many data association errors and recover observations from an occluded state. We demonstrate the robustness of our data-driven approach by tracking multiple targets using their appearance, motion, and even interactions. Our method outperforms previous works on multiple publicly available datasets including the challenging MOT benchmark.

[paper](https://arxiv.org/abs/1701.01909)

### 2016

#### Person Re-Identification by Multi-Channel Parts-Based CNN with Improved Triplet Loss Function

Person re-identification across cameras remains a very
challenging problem, especially when there are no overlapping fields of view between cameras. In this paper,
we present a novel multi-channel parts-based convolutional neural network (CNN) model under the triplet framework
for person re-identification. Specifically, the proposed CNN
model consists of multiple channels to jointly learn both the
global full-body and local body-parts features of the input
persons. The CNN model is trained by an improved triplet
loss function that serves to pull the instances of the same
person closer, and at the same time push the instances belonging to different persons farther from each other in the
learned feature space. Extensive comparative evaluations demonstrate that our proposed method significantly outperforms many state-of-the-art approaches, including both
traditional and deep network-based ones, on the challenging i-LIDS, VIPeR, PRID2011 and CUHK01 datasets.

[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Cheng_Person_Re-Identification_by_CVPR_2016_paper.pdf)

#### Learning by tracking: Siamese CNN for robust target association

This paper introduces a novel approach to the task of data association within the context of pedestrian tracking, by introducing a two-stage learning scheme to match pairs of detections. First, a Siamese convolutional neural network (CNN) is trained to learn descriptors encoding local spatio-temporal structures between the two input image patches, aggregating pixel values and optical flow information. Second, a set of contextual features derived from the position and size of the compared input patches are combined with the CNN output by means of a gradient boosting classifier to generate the final matching probability. This learning approach is validated by using a linear programming based multi-person tracker showing that even a simple and efficient tracker may outperform much more complex models when fed with our learned matching probabilities. Results on publicly available sequences show that our method meets state-of-the-art standards in multiple people tracking.

[paper](https://arxiv.org/abs/1604.07866)

#### SORT

This paper explores a pragmatic approach to multiple object tracking where the main focus is to associate objects efficiently for online and realtime applications. To this end, detection quality is identified as a key factor influencing tracking performance, where changing the detector can improve tracking by up to 18.9%. Despite only using a rudimentary combination of familiar techniques such as the Kalman Filter and Hungarian algorithm for the tracking components, this approach achieves an accuracy comparable to state-of-the-art online trackers. Furthermore, due to the simplicity of our tracking method, the tracker updates at a rate of 260 Hz which is over 20x faster than other state-of-the-art trackers.

[paper](https://arxiv.org/abs/1602.00763)

[python code](https://github.com/abewley/sort)




## Joint Tracking and ID

Joint tracking-as-ID integrates object detection and tracking into a single framework, viewing the tracking task as an end-to-end ID assignment problem. Instead of associating detections post hoc, it predicts both object identities (IDs) and trajectories directly from video frames.

key concepts: End-to-End Training, ID Prediction as Contextual Prompting, Transformer and GNN Architectures, Applications to Any Object.

There are no handcrafted heuristics, uses global optimization, and usually generalizes very well to unseen scenarios, occlusion, and object interactions.


### 2024

#### Multiple Object Tracking as ID Prediction

In Multiple Object Tracking (MOT), tracking-by-detection methods have stood the test for a long time, which split the process into two parts according to the definition: object detection and association. They leverage robust single-frame detectors and treat object association as a post-processing step through hand-crafted heuristic algorithms and surrogate tasks. However, the nature of heuristic techniques prevents end-to-end exploitation of training data, leading to increasingly cumbersome and challenging manual modification while facing complicated or novel scenarios. In this paper, we regard this object association task as an End-to-End in-context ID prediction problem and propose a streamlined baseline called MOTIP. Specifically, we form the target embeddings into historical trajectory information while considering the corresponding IDs as in-context prompts, then directly predict the ID labels for the objects in the current frame. Thanks to this end-to-end process, MOTIP can learn tracking capabilities straight from training data, freeing itself from burdensome hand-crafted algorithms. Without bells and whistles, our method achieves impressive state-of-the-art performance in complex scenarios like DanceTrack and SportsMOT, and it performs competitively with other transformer-based methods on MOT17. We believe that MOTIP demonstrates remarkable potential and can serve as a starting point for future research

[paper](https://arxiv.org/abs/2403.16848)

[python code](https://github.com/MCG-NJU/MOTIP)

### 2022

#### Global Tracking Transformers

We present a novel transformer-based architecture for
global multi-object tracking. Our network takes a short
sequence of frames as input and produces global trajectories for all objects. The core component is a global tracking transformer that operates on objects from all frames
in the sequence. The transformer encodes object features
from all frames, and uses trajectory queries to group them
into trajectories. The trajectory queries are object features
from a single frame and naturally produce unique trajectories. Our global tracking transformer does not require
intermediate pairwise grouping or combinatorial association, and can be jointly trained with an object detector. It
achieves competitive performance on the popular MOT17
benchmark, with 75.3 MOTA and 59.1 HOTA. More importantly, our framework seamlessly integrates into stateof-the-art large-vocabulary detectors to track any objects.
Experiments on the challenging TAO dataset show that our
framework consistently improves upon baselines that are
based on pairwise association, outperforming published
work by a significant 7.7 tracking mAP

[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Global_Tracking_Transformers_CVPR_2022_paper.pdf)

[python code](https://github.com/xingyizhou/GTR)

### 2021

#### Joint Object Detection and Multi-Object Tracking with Graph Neural Networks

Object detection and data association are critical components in multi-object tracking (MOT) systems. Despite the fact that the two components are dependent on each other, prior works often design detection and data association modules separately which are trained with separate objectives. As a result, one cannot back-propagate the gradients and optimize the entire MOT system, which leads to sub-optimal performance. To address this issue, recent works simultaneously optimize detection and data association modules under a joint MOT framework, which has shown improved performance in both modules. In this work, we propose a new instance of joint MOT approach based on Graph Neural Networks (GNNs). The key idea is that GNNs can model relations between variable-sized objects in both the spatial and temporal domains, which is essential for learning discriminative features for detection and data association. Through extensive experiments on the MOT15/16/17/20 datasets, we demonstrate the effectiveness of our GNN-based joint MOT approach and show state-of-the-art performance for both detection and MOT tasks.

[paper](https://arxiv.org/abs/2006.13164)

[python code](https://github.com/yongxinw/GSDT)

#### Track to Detect and Segment: An Online Multi-Object Tracker

Most online multi-object trackers perform object detection stand-alone in a neural net without any input from tracking. In this paper, we present a new online joint detection and tracking model, TraDeS (TRAck to DEtect and Segment), exploiting tracking clues to assist detection end-to-end. TraDeS infers object tracking offset by a cost volume, which is used to propagate previous object features for improving current object detection and segmentation. Effectiveness and superiority of TraDeS are shown on 4 datasets, including MOT (2D tracking), nuScenes (3D tracking), MOTS and Youtube-VIS (instance segmentation tracking)

[paper](https://arxiv.org/abs/2103.08808)

[python code](https://github.com/JialianW/TraDeS)

### 2020


#### TransTrack: Multiple Object Tracking with Transformer

In this work, we propose TransTrack, a simple but efficient scheme to solve the multiple object tracking problems. TransTrack leverages the transformer architecture, which is an attention-based query-key mechanism. It applies object features from the previous frame as a query of the current frame and introduces a set of learned object queries to enable detecting new-coming objects. It builds up a novel joint-detection-and-tracking paradigm by accomplishing object detection and object association in a single shot, simplifying complicated multi-step settings in tracking-by-detection methods. On MOT17 and MOT20 benchmark, TransTrack achieves 74.5\% and 64.5\% MOTA, respectively, competitive to the state-of-the-art methods. We expect TransTrack to provide a novel perspective for multiple object tracking.

[paper](https://arxiv.org/abs/2012.15460)

[python code](https://github.com/PeizeSun/TransTrack)

#### Learning a Neural Solver for Multiple Object Tracking

Graphs offer a natural way to formulate Multiple Object Tracking (MOT) within the tracking-by-detection paradigm. However, they also introduce a major challenge for learning methods, as defining a model that can operate on such \textit{structured domain} is not trivial. As a consequence, most learning-based work has been devoted to learning better features for MOT, and then using these with well-established optimization frameworks. In this work, we exploit the classical network flow formulation of MOT to define a fully differentiable framework based on Message Passing Networks (MPNs). By operating directly on the graph domain, our method can reason globally over an entire set of detections and predict final solutions. Hence, we show that learning in MOT does not need to be restricted to feature extraction, but it can also be applied to the data association step. We show a significant improvement in both MOTA and IDF1 on three publicly available benchmarks

[paper](https://arxiv.org/abs/1912.07515)

[python code](https://github.com/dvl-tum/mot_neural_solver)


#### RetinaTrack: Online Single Stage Joint Detection and Tracking

Traditionally multi-object tracking and object detection are performed using separate systems with most prior works focusing exclusively on one of these aspects over the other. Tracking systems clearly benefit from having access to accurate detections, however and there is ample evidence in literature that detectors can benefit from tracking which, for example, can help to smooth predictions over time. In this paper we focus on the tracking-by-detection paradigm for autonomous driving where both tasks are mission critical. We propose a conceptually simple and efficient joint model of detection and tracking, called RetinaTrack, which modifies the popular single stage RetinaNet approach such that it is amenable to instance-level embedding training. We show, via evaluations on the Waymo Open Dataset, that we outperform a recent state of the art tracking algorithm while requiring significantly less computation. We believe that our simple yet effective approach can serve as a strong baseline for future work in this area.


[paper](https://arxiv.org/abs/2003.13870)

[pytorch code](https://github.com/Hanson0910/RetinaTrack)

#### TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model

Multi-object tracking is a fundamental vision problem that has been studied for a long time. As deep learning brings excellent performances to object detection algorithms, Tracking by Detection (TBD) has become the mainstream tracking framework. Despite the success of TBD, this two-step method is too complicated to train in an end-to-end manner and induces many challenges as well, such as insufficient exploration of video spatial-temporal information, vulnerability when facing object occlusion, and excessive reliance on detection results. To address these challenges, we propose a concise end-to-end model TubeTK which only needs one step training by introducing the "bounding-tube" to indicate temporal-spatial locations of objects in a short video clip. TubeTK provides a novel direction of multi-object tracking, and we demonstrate its potential to solve the above challenges without bells and whistles. We analyze the performance of TubeTK on several MOT benchmarks and provide empirical evidence to show that TubeTK has the ability to overcome occlusions to some extent without any ancillary technologies like Re-ID. Compared with other methods that adopt private detection results, our one-stage end-to-end model achieves state-of-the-art performances even if it adopts no ready-made detection results. We hope that the proposed TubeTK model can serve as a simple but strong alternative for video-based MOT task. 

[paper](https://arxiv.org/abs/2006.05683)

[python code](https://github.com/BoPang1996/TubeTK)

#### Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking

Existing Multiple-Object Tracking (MOT) methods either follow the tracking-by-detection paradigm to conduct object detection, feature extraction and data association separately, or have two of the three subtasks integrated to form a partially end-to-end solution. Going beyond these sub-optimal frameworks, we propose a simple online model named Chained-Tracker (CTracker), which naturally integrates all the three subtasks into an end-to-end solution (the first as far as we know). It chains paired bounding boxes regression results estimated from overlapping nodes, of which each node covers two adjacent frames. The paired regression is made attentive by object-attention (brought by a detection module) and identity-attention (ensured by an ID verification module). The two major novelties: chained structure and paired attentive regression, make CTracker simple, fast and effective, setting new MOTA records on MOT16 and MOT17 challenge datasets (67.6 and 66.6, respectively), without relying on any extra training data.

[paper](https://arxiv.org/abs/2007.14557)

[python code](https://github.com/pjl1995/CTracker)

### 2019

#### Tracking without bells and whistles

The problem of tracking multiple objects in a video sequence poses several challenging tasks. For tracking-by detection, these include object re-identification, motion prediction and dealing with occlusions. We present a tracker
(without bells and whistles) that accomplishes tracking
without specifically targeting any of these tasks, in particular, we perform no training or optimization on tracking
data. To this end, we exploit the bounding box regression of
an object detector to predict the position of an object in the
next frame, thereby converting a detector into a Tracktor.
We demonstrate the potential of Tracktor and provide a new
state-of-the-art on three multi-object tracking benchmarks
by extending it with a straightforward re-identification and
camera motion compensation.
We then perform an analysis on the performance and
failure cases of several state-of-the-art tracking methods
in comparison to our Tracktor. Surprisingly, none of the
dedicated tracking methods are considerably better in dealing with complex tracking scenarios, namely, small and
occluded objects or missing detections. However, our approach tackles most of the easy tracking scenarios. Therefore, we motivate our approach as a new tracking paradigm
and point out promising future research directions. Overall, Tracktor yields superior tracking performance than any
current tracking method and our analysis exposes remaining and unsolved tracking challenges to inspire future research directions.

[paper](https://arxiv.org/pdf/1903.05625)

[python code](https://github.com/phil-bergmann/tracking_wo_bnw)





## End-to-End 



### 2022

#### MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors

In this paper, we propose MOTRv2, a simple yet effective pipeline to bootstrap end-to-end multi-object tracking with a pretrained object detector. Existing end-to-end methods, MOTR and TrackFormer are inferior to their tracking-by-detection counterparts mainly due to their poor detection performance. We aim to improve MOTR by elegantly incorporating an extra object detector. We first adopt the anchor formulation of queries and then use an extra object detector to generate proposals as anchors, providing detection prior to MOTR. The simple modification greatly eases the conflict between joint learning detection and association tasks in MOTR. MOTRv2 keeps the query propogation feature and scales well on large-scale benchmarks. MOTRv2 ranks the 1st place (73.4% HOTA on DanceTrack) in the 1st Multiple People Tracking in Group Dance Challenge. Moreover, MOTRv2 reaches state-of-the-art performance on the BDD100K dataset. We hope this simple and effective pipeline can provide some new insights to the end-to-end MOT community.

[paper](https://arxiv.org/abs/2211.09791)

[python code](https://github.com/megvii-research/MOTRv2?tab=readme-ov-file)

#### TrackFormer

The challenging task of multi-object tracking (MOT) requires simultaneous reasoning about track initialization, identity, and spatio-temporal trajectories. We formulate this task as a frame-to-frame set prediction problem and introduce TrackFormer, an end-to-end trainable MOT approach based on an encoder-decoder Transformer architecture. Our model achieves data association between frames via attention by evolving a set of track predictions through a video sequence. The Transformer decoder initializes new tracks from static object queries and autoregressively follows existing tracks in space and time with the conceptually new and identity preserving track queries. Both query types benefit from self- and encoder-decoder attention on global frame-level features, thereby omitting any additional graph optimization or modeling of motion and/or appearance. TrackFormer introduces a new tracking-by-attention paradigm and while simple in its design is able to achieve state-of-the-art performance on the task of multi-object tracking (MOT17 and MOT20) and segmentation (MOTS20).

[paper](https://arxiv.org/abs/2101.02702)

[python code](https://github.com/timmeinhardt/trackformer.)


### 2021

#### MOTR: End-to-End Multiple-Object Tracking with Transformer

Temporal modeling of objects is a key challenge in multiple object tracking (MOT). Existing methods track by associating detections through motion-based and appearance-based similarity heuristics. The post-processing nature of association prevents end-to-end exploitation of temporal variations in video sequence. In this paper, we propose MOTR, which extends DETR and introduces track query to model the tracked instances in the entire video. Track query is transferred and updated frame-by-frame to perform iterative prediction over time. We propose tracklet-aware label assignment to train track queries and newborn object queries. We further propose temporal aggregation network and collective average loss to enhance temporal relation modeling. Experimental results on DanceTrack show that MOTR significantly outperforms state-of-the-art method, ByteTrack by 6.5% on HOTA metric. On MOT17, MOTR outperforms our concurrent works, TrackFormer and TransTrack, on association performance. MOTR can serve as a stronger baseline for future research on temporal modeling and Transformer-based trackers.

[paper](https://arxiv.org/abs/2105.03247)

[python code](https://github.com/megvii-research/MOTR)

#### Learning to Track with Object Permanence

Tracking by detection, the dominant approach for online multi-object tracking, alternates between localization and association steps. As a result, it strongly depends on the quality of instantaneous observations, often failing when objects are not fully visible. In contrast, tracking in humans is underlined by the notion of object permanence: once an object is recognized, we are aware of its physical existence and can approximately localize it even under full occlusions. In this work, we introduce an end-to-end trainable approach for joint object detection and tracking that is capable of such reasoning. We build on top of the recent CenterTrack architecture, which takes pairs of frames as input, and extend it to videos of arbitrary length. To this end, we augment the model with a spatio-temporal, recurrent memory module, allowing it to reason about object locations and identities in the current frame using all the previous history. It is, however, not obvious how to train such an approach. We study this question on a new, large-scale, synthetic dataset for multi-object tracking, which provides ground truth annotations for invisible objects, and propose several approaches for supervising tracking behind occlusions. Our model, trained jointly on synthetic and real data, outperforms the state of the art on KITTI and MOT17 datasets thanks to its robustness to occlusions.

[paper](https://arxiv.org/abs/2103.14258)


### 2020


#### Simple Unsupervised Multi-Object Tracking

Multi-object tracking has seen a lot of progress recently, albeit with substantial annotation costs for developing better and larger labeled datasets. In this work, we remove the need for annotated datasets by proposing an unsupervised re-identification network, thus sidestepping the labeling costs entirely, required for training. Given unlabeled videos, our proposed method (SimpleReID) first generates tracking labels using SORT and trains a ReID network to predict the generated labels using crossentropy loss. We demonstrate that SimpleReID performs substantially better than simpler alternatives, and we recover the full performance of its supervised counterpart consistently across diverse tracking frameworks. The observations are unusual because unsupervised ReID is not expected to excel in crowded scenarios with occlusions, and drastic viewpoint changes. By incorporating our unsupervised SimpleReID with CenterTrack trained on augmented still images, we establish a new state-of-the-art performance on popular datasets like MOT16/17 without using tracking supervision, beating current best (CenterTrack) by 0.2-0.3 MOTA and 4.4-4.8 IDF1 scores. We further provide evidence for limited scope for improvement in IDF1 scores beyond our unsupervised ReID in the studied settings. Our investigation suggests reconsideration towards more sophisticated, supervised, end-to-end trackers by showing promise in simpler unsupervised alternatives.

[paper](https://arxiv.org/abs/2006.02609)

### 2019

#### How To Train Your Deep Multi-Object Tracker

The recent trend in vision-based multi-object tracking (MOT) is heading towards leveraging the representational power of deep learning to jointly learn to detect and track objects. However, existing methods train only certain sub-modules using loss functions that often do not correlate with established tracking evaluation measures such as Multi-Object Tracking Accuracy (MOTA) and Precision (MOTP). As these measures are not differentiable, the choice of appropriate loss functions for end-to-end training of multi-object tracking methods is still an open research problem. In this paper, we bridge this gap by proposing a differentiable proxy of MOTA and MOTP, which we combine in a loss function suitable for end-to-end training of deep multi-object trackers. As a key ingredient, we propose a Deep Hungarian Net (DHN) module that approximates the Hungarian matching algorithm. DHN allows estimating the correspondence between object tracks and ground truth objects to compute differentiable proxies of MOTA and MOTP, which are in turn used to optimize deep trackers directly. We experimentally demonstrate that the proposed differentiable framework improves the performance of existing multi-object trackers, and we establish a new state of the art on the MOTChallenge benchmark.

[paper](https://arxiv.org/abs/1906.06618)

[python code](https://github.com/yihongXU/deepMOT)


## 3D Tracking

### 2024

#### MCTrack: A Unified 3D Multi-Object Tracking Framework for Autonomous Driving

This paper introduces MCTrack, a new 3D multi-object tracking method that achieves state-of-the-art (SOTA) performance across KITTI, nuScenes, and Waymo datasets. Addressing the gap in existing tracking paradigms, which often perform well on specific datasets but lack generalizability, MCTrack offers a unified solution. Additionally, we have standardized the format of perceptual results across various datasets, termed BaseVersion, facilitating researchers in the field of multi-object tracking (MOT) to concentrate on the core algorithmic development without the undue burden of data preprocessing. Finally, recognizing the limitations of current evaluation metrics, we propose a novel set that assesses motion information output, such as velocity and acceleration, crucial for downstream tasks.

[paper](https://arxiv.org/abs/2409.16149)

[python code](https://github.com/megvii-research/MCTrack)


## Animal Tracking / Implementation

#### An HMM-based framework for identity-aware long-term multi-object tracking from sparse and uncertain identification: use case on long-term tracking in livestock

[paper](https://drive.google.com/file/d/1_-6oLD4X2FHp3bo-Qp4PDtcpEMWr0kIL/view)

[python code](https://github.com/ngobibibnbe/uncertain-identity-aware-tracking)