### Acknowledgments 

We would like to thank the anonymous reviewers and (S)ACs of ICML 2023 for their constructive comments and dedicated service to the community. We also would like to thank all reviewers’ engagement during the author-reviewer discussion period.

----

### Meta Review by Area Chair

Dear Author(s),

We appreciate your submission to ICML2023 on Omni-generalizable Neural Methods for VRP. 

Your paper underwent a thorough review process, during which the committee evaluated the contribution of your work. We are pleased to announce that the paper was accepted to ICML.

While we recognize the amount of time and effort you have invested in your research, the reviewers identified several issues in your paper that need to be addressed in your camera-ready version of the paper. We encourage authors to take into account feedback from reviewers in order to improve their paper. 

Sincerely, AC

----

### Official Review by Reviewer iBeg

**Summary:**

The paper proposes a meta-learning framework for solving vehicle routing problems, which generalizes well to unseen problem sizes and distributions. The approach is based on a hierarchical scheduler that gradually samples new training tasks. Experiments are conducted in a zero-shot setting and a few-shot setting, in which a small set of instances from the target distribution/size are used for fine-tuning. The performance of the model compares favorably with previous baselines (variants of POMO).

**Strengths And Weaknesses:**

This paper addresses an important issue of existing neural methods for vehicle routing problems, namely the generalization to problem instances of unseen size and distributions. It also permits significant enhancements to the performance of a previous architecture, albeit through a more convoluted training process. The experimental setting is convincing and comprehensive, considering well-known baseline methods and algorithms. On the other hand, the methodological contribution of the paper is a bit more limited, as it focuses on training enhancement of existing architectures and does not include theoretical results backing up the experimental observations.

**Questions:**

As a weakness, I noted that the generalization capabilities to instances with unseen distributions (in the paper) are focused on cases with different distributions governing customers' locations. Another critical aspect of vehicle routing problem instances is the distribution of customer demands, which can differ a lot between applications. Benchmark Instances with a broader range of demand distributions, customer location, and depot-location distributions have been presented in [Queiroga et al. (2022). 10,000 optimal CVRP solutions for testing machine learning-based heuristics. In AAAI-22 Workshop on Machine Learning for Operations Research.] and would be a good benchmark for testing generalization capabilities on a less restricted distribution of instances (that more closely matches the original Set X distribution).

**Post Rebuttal:** Dear authors, Thank you for thoroughly considering my recommendations and conducting these additional experiments, which will definitely add value to the paper by introducing datasets with a more diverse distribution. I still view the contribution as more on the experimental and practical side, in contrast with the usual methodological and theoretical contributions seen in conferences such as ICML. Given these reserves, I still raise my score to acknowledge that some of my initial concerns have been addressed and let the final decision regarding the importance of the methodological contribution be in the hands of the area chairs.

**Limitations:**

I do not identify a potential negative societal impact connected to this work.

**Ethics Flag:** No

**Soundness:** 3 good

**Presentation:** 2 fair

**Contribution:** 2 fair

**Rating:** 5 -> 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.

**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Code Of Conduct:** Yes

----

### Official Review by Reviewer H48e

**Summary:**

This paper focuses on generalizing learned heuristics across size and distributions in VRPs. It proposes using the meta-learning framework to achieve fast adaptation of POMO (2020) to VRPs with new distribution or sizes. The method first learns the initialization of model parameters by performing meta-training on several VRP tasks and then adapts the model to new tasks by gradient descent only using limited data during inference. The standard meta-learning framework is modified by a hierarchical task scheduler and an improved second-order derivative approximation. The results on TSP and CVRP show the method could generalize well on few-shot or zero-shot settings. The contributions of the paper are using standard meta-learning to improve the generalization of POMO across size and distribution.

**Strengths And Weaknesses:**

Strengths:

1. The topic of generalizing learned VRP heuristics is important and interesting. 
2. The proposed meta-learning framework seems reasonable.
3. The paper is mostly well-written. The empirical results show that the paper is able to achieve better results compared to some existing solvers.

Weaknesses:

1. The meta-learning framework is mostly from the standard MAML (Finn 2017). The modifications about the hierarchical task scheduler and improved second-order derivative approximation are a bit incremental and marginal.
2. The results of the proposed method are not significant compared with Meta-POMO. The gap difference is usually less than 0.5%. 
3. The method is only generalized to VRP 1000. The reviewer encourages the authors to validate the method on larger-scale TSP or VRP from VRPLIB.
4. Some recent relevant works from ML and OR fields are missing in the reviews or not compared, such as the Learn-to-delegate (Li, 2021), TAM (Hou, ICLR2023), Learn-to-improve (Lu, 2019) from ML field, and SISR for large-scale VRPs from the OR field.
5. Necessary ablation studies are welcomed.

**Post Rebuttal:** Thanks for the detailed responses, more experiments, and ablation studies. I think this paper is a good application of MAML on VRP. However, the technical contribution compared with MAML and Meta-POMO is marginal. Given the new results in rebuttal, the gap between the proposed method and Meta-POMO is still not significant. Therefore, I raise my score to 5 but keep it borderline.

**Questions:**

Please see the weaknesses.

**Limitations:**

The limitations of the work are not reported or discussed in the paper.

**Ethics Flag:** No

**Soundness:** 2 fair

**Presentation:** 3 good

**Contribution:** 2 fair

**Rating:** 4 -> 5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Code Of Conduct:** Yes

----

### Official Review by Reviewer NCRY

**Summary:**

This paper introduces a new approach to combinatorial optimization that employs a meta-learning framework. The proposed method is based on the state-of-the-art CO model POMO, developed by Kwon et al., and takes into account the scalability and robustness of the distribution. Drawing inspiration from successful meta-learning techniques like MAML, the authors propose a model-agnostic gradient-based meta-learning approach.

To distribute tasks effectively, the authors use problem distribution hardness, such as uniform or clustered distribution, and a curriculum-based approach to scale increment. The authors apply second-order MAML to train POMO under the predefined task distribution, demonstrating its effectiveness in handling distributional shifts. Compared to recent studies that focus on the generalization capabilities of neural combinatorial optimization models (e.g., Meta-POMO), this method outperforms them in terms of performance.

[1] Kwon, Yeong-Dae, et al. "Pomo: Policy optimization with multiple optima for reinforcement learning." Advances in Neural Information Processing Systems 33 (2020): 21188-21198.

[2] Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." International conference on machine learning. PMLR, 2017.

**Strengths And Weaknesses:**

**Strengths**

This paper presents a natural application of meta-learning to combinatorial optimization. Since learning a neural solver for combinatorial optimization can be viewed as contextual multi-task reinforcement learning, meta-learning techniques can offer an effective way to improve generalization performance. While the POMO model performs exceptionally well on specific distributions, such as the uniform distribution of TSP (N=100), it suffers from significant limitations when dealing with distributional shifts. This method improves the robustness and generalization capability of POMO by applying a simple meta-learning procedure.

The primary contribution of this paper is the introduction of a novel task distribution approach, which defines the instance distribution and problem scale based on their level of difficulty. The proposed method leverages MAML, a state-of-the-art meta-learning technique, to achieve these improvements. Overall, this paper offers valuable insights into how meta-learning can be used to enhance the performance and robustness of combinatorial optimization models.

**Weaknesses**

1. Training efficiency.

It is worth noting that POMO, the CO model used in this paper, already requires substantial training time. However, this proposed meta-learning method also requires a significant amount of training time, taking 5 days with a 53GB GPU. It raises concerns that learning combinatorial optimization with current methods may not be efficient or practical. The research community needs to focus on developing more efficient algorithms that can handle larger-scale problems. For instance, if we want to solve TSP with 10,000 nodes or CVRP with 100,000 nodes, the scalability of the proposed algorithm is questionable, and there is a need for more robust and scalable solutions.

2. Lack of analysis

I think the paper should contain more analysis of why their method is better than prior methods. For example, I'm curious about the sensitivity of performances on task distribution. 

3. In-distribution vs. Out-of-distribution.

This paper is focused on meta-learning techniques and their applicability to combinatorial optimization problems. It is suggested that the paper should include research on out-of-distribution training task distributions. While the authors did provide results on out-of-distribution tasks regarding the scale of nodes, testing on larger scales than those used for training, the results are not entirely convincing. Their method appears to perform poorly in out-of-distribution settings, which warrants further investigation.

4. Meta-learning

This paper applies MAML, a meta-learning technique that is known for its training instability and computational complexity, to combinatorial optimization problems. However, given the complexity of CO problems, it is essential for researchers in this field to design their own specialized meta-learning algorithms. Therefore, the technical novelty of this paper is limited since it relies on a well-established technique rather than developing a new, CO-specific approach. Nonetheless, this paper provides valuable insights into how meta-learning can be applied to CO problems, and its findings can inspire future research into developing more efficient and effective meta-learning techniques for CO.

------

Despite several limitations, I give borderline accept because I like the motivation of the paper, and I know how difficult to train a neural combinatorial optimization model using meta-learning from a technical point of view. 

------

**Post Rebuttal:** I appreciate that the authors tried to convince my concerns during the rebuttal phase. Most concerns were convinced, but I still think this work is borderline level. I just increased my score to 6, but I decreased my confidence.

**Questions:**

1. Can your method apply to pretrained POMO? I think that can make it time-saving. 
2. How sensible that your method on hyperparameters? 
3. Do you plan to release reproducible codes? 
4. Can your method be able to train N>300? It seems that 13.38% optimal gap is too high at TSP (N=1000). That's because other meta-learning for combinatorial optimization methods [1] gives 3.19% optimal gap at TSP (N=10000).

Minor Suggestion:

The table is too small. Please consider the presentation of your experimental results in terms of the size of the fonts. Maybe you can leverage the graph for presents your main results. 

[1] Qiu, Ruizhong, Zhiqing Sun, and Yiming Yang. "DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems." arXiv preprint arXiv:2210.04123 (2022).

**Limitations:**

Technical Limitations:

1. Training Scalability.
2. Performances on large scales.

Social Limitations:

None.

**Ethics Flag:** No

**Soundness:** 3 good

**Presentation:** 2 fair

**Contribution:** 2 fair

**Rating:** 5 -> 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.

**Confidence:** 4 -> 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Code Of Conduct:** Yes

----

### Official Review by Reviewer hRux

**Summary:**

The authors of this study investigate a realistic and challenging scenario for neural methods in the Vehicle Routing Problem (VRP), which involves achieving omni-generalization across diverse sizes and distributions of the problem. To address this, they introduce a meta-learning framework capable of efficiently adapting to new tasks using limited data during inference. The authors further develop a simple approximation method and assess its efficacy on both the Traveling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP). The experimental results demonstrate the method's potential to enhance the generalization of the base model.

**Strengths And Weaknesses:**

Strengths:

1. Well-organized paper structure.
2. Thorough review of related work.
3. The proposed meta-learning framework has potential practical applications.
4. The method improves the performance of the POMO inference strategy.
5. The authors provide code and sufficient information for reproducibility.

Weaknesses:

1. The baseline used is not a state-of-the-art RL implementation for VRP, e.g. (Lu, Hao, 2020) shows better performance on VRP than the LKH3 heuristic.
2. The generalizability of the proposed method to other base models is unclear, as only the POMO model was tested.
3. The observed improvement in performance is relatively small.
4. The overall impact of the proposed method on the other base models is unclear.

**Questions:**

1. Have you considered benchmarking against other state-of-the-art RL methods that provide better results than LKH3 and Concorde heuristics?
2. Have you tried to check the true optimality gap for VRP instances with less than 30 nodes? 
3. Have you considered evaluating your method on larger problem sizes that are commonly encountered in the industry, such as VRPs with thousands of nodes? It would be interesting to see how your proposed method scales with problem size.

It is conjectured that if the authors empirically demonstrate the superiority of their proposed method on models other than POMO, it could significantly enhance the paper's strength.

**Post Rebuttal:** Thank you for your solid and detailed responses. You provided sufficient answers to my questions. I raise my score to 6.

**Limitations:**

No limitations or potential negative social impact of this work at this stage.

**Ethics Flag:** No

**Soundness:** 3 good

**Presentation:** 3 good

**Contribution:** 2 fair

**Rating:** 4 -> 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Code Of Conduct:** Yes