---
license: mit
---
## Abstract

We introduce a foundation model for event classification in high-energy physics, built on a **Graph Neural Network** architecture and trained on **120 million simulated proton-proton collision events** spanning 12 distinct physics processes. The model is *pretrained* to learn a general and robust representation of collision data using challenging multiclass and multilabel classification tasks.

Its performance is evaluated across five event classification tasks, which include both physics processes used during pretraining and new processes not encountered during pretraining. Fine-tuning the pretrained model significantly improves classification performance, particularly in scenarios with limited training data, demonstrating gains in both accuracy and computational efficiency.

To investigate the underlying mechanisms behind these performance improvements, we employ a representational similarity evaluation framework based on *Centered Kernel Alignment*. This analysis reveals notable differences in the learned representations of fine-tuned pretrained models compared to baseline models trained from scratch.

## Introduction

Machine learning has become a ubiquitous tool in particle physics, employed in a variety of tasks including triggering, simulation, reconstruction, and offline analysis. While its utility spans classification, regression, and generative tasks, the current paradigm of developing machine learning models from scratch for each specific application presents several challenges. This approach not only demands specialized expertise and substantial computing resources but can also result in suboptimal performance due to limited training data. The from-scratch development of models necessitates individual validation studies to ensure that neural networks utilize well-modeled information from training samples, whether derived from Monte Carlo simulations or control samples from experimental data.

Foundation models offer a promising direction to address these limitations. These models, pre-trained on large, diverse datasets across various tasks, provide robust and general representations of underlying data structures. Notable examples in other fields include GPT-4 [OpenAI et al., 2024](#ref-openai-2024-gpt4) and BERT [Devlin et al., 2018](#ref-devlin-2018-bert) in natural language processing, Stable Diffusion [Rombach et al., 2021](#ref-rombach-2021-latentdiffusion) in image processing, and AlphaFold [Jumper et al., 2021](#ref-jumper-2021-alphafold) in structural biology. The foundation model approach offers several advantages for particle physics applications: reduced computing resources for fine-tuning [Yosinski et al., 2014](#ref-yosinski-2014-transfer) compared to training from scratch, superior performance on specific tasks (particularly with limited training data), and potentially simplified validation procedures as downstream tasks inherit verified representations from the pre-trained model.

Current literature on pretrained models for particle physics can be categorized based on the data representation they handle. Models operating on particle- or event-level numerical data use features like particle four momenta or jets, leveraging self-supervised or generative methods to learn versatile representations. Detector-focused models operate on high-dimensional responses such as calorimeter deposits or pixel hits, employing geometry-aware techniques for accurate simulation and analysis. Finally, models using textual or code representations apply large language model architectures to integrate domain knowledge, enabling tasks like question answering and code generation.

Recent studies have begun exploring foundation models tailored to particle physics data, which has a variety of distinct structures and properties across many experiments and data processing stages, including:

- particle-level & event-level numeric data [Wildridge et al., 2024](#ref-wildridge-2024-bumblebee), [Katel et al., 2024](#ref-katel-2024-jet), [Golling et al., 2024](#ref-golling-2024-maskedset), [Mikuni & Nachman, 2024](#ref-mikuni-2024-omnilearn), [Harris et al., 2024](#ref-harris-2024-resimulation), [Birk et al., 2024](#ref-birk-2024-omnijet), [Vigl et al., 2024](#ref-vigl-2024-finetune),
- detector-level & geometry-aware data [Araz et al., 2024](#ref-araz-2024-pointcloud), [Liu et al., 2023](#ref-liu-2023-gaam), [Hashemi et al., 2024](#ref-hashemi-2024-gen), [Huang et al., 2024](#ref-huang-2024-lmtracking),
- textual or code data [Zhang et al., 2024](#ref-zhang-2024-xiwu).

This paper presents a foundation model designed specifically for collider event-level data. In modern collider experiments, final-stage analysis processes information from reconstructed objects that either directly correspond to particles in collision final states (such as leptons and photons) or serve as proxies (such as jets and missing transverse energy). While traditional approaches often relied on "high-level" variables calculated from object features, recent trends favor direct input of event objects and their features into neural networks for analysis tasks. A notable example is [ATLAS Collaboration, 2023](#ref-atlas-2023-4top), which established the observation of simultaneous production of four top quarks with the ATLAS experiment by employing a graph neural network (GNN) architecture to process event-level object information.

We present foundation models that adopt an architecture similar to that used for [ATLAS Collaboration, 2023](#ref-atlas-2023-4top). Our models are pre-trained using either multiclass classification or multi-label learning tasks across 12 distinct physics processes. We evaluate these models through fine-tuning and testing on five classification tasks, including both familiar and novel processes not seen during pre-training. Our analysis benchmarks the models' performance improvements, their scaling behavior with training sample size, and computational efficiency, representing the first prototype of a foundation model operating on collider final-state object data.

## Data Samples

To provide a diverse set of physics processes for the pretraining, we use Madgraph@NLO 2.7.3 [Alwall et al., 2014](#ref-alwall-2014hca) to generate proton-proton collision events at next-to-leading order (NLO) in Quantum Chromodynamics (QCD). We generate 12 distinct Standard Model (SM) physics processes, including six major Higgs boson production mechanisms: gluon fusion production \\(ggF\\), vector boson fusion \\(VBF\\), associated production of the Higgs boson with a W boson \\(WH\\) or a Z boson \\(ZH\\), associated production of the Higgs boson with a top-quark pair \\(t\bar{t}H\\), and associated production of the Higgs boson with a single top quark and a forward quark \\(tHq\\). Additionally, we simulate six top quark production processes: single top production, top-quark pair production \\(t\bar{t}\\), top quark pair production in association with a pair of photons \\(t\bar{t}\gamma\gamma\\), associated production of a top-quark pair with a W boson \\(t\bar{t}W\\), simultaneous production of three top quarks \\(t\bar{t}t\\), and simultaneous production of four top quarks \\(t\bar{t}t\bar{t}\\). In these samples, the Higgs boson and top quarks decay inclusively. These 12 Higgs and top quark production processes constitute the pretraining dataset.

To test the pretrained model, we further generated four processes including three beyond Standard Model (SM) processes: a SM \\(t\\bar{t}H\\) production where the Higgs boson decays exclusively to a pair of photons, a \\(t\\bar{t}H\\) production with the Higgs boson decaying to a pair of photons, where the top-Yukawa coupling is CP-odd, implemented using the Higgs Characterization model [Artoisenet et al., 2013](#ref-artoisinet-2013puc), the production of a pair of superpartners of the top quark (s-top) using the Minimal Supersymmetric Standard Model (MSSM) [Rosiek, 1990](#ref-rosiek-1990), [Allanach et al., 2009](#ref-allanach-2009), and flavor changing neutral current (FCNC) processes [Degrande et al., 2015](#ref-degrande-2015), [Durieux et al., 2015](#ref-durieux-2015). For the s-top process, we simulate the production of heavier s-top pairs \\(t_2\\bar{t_2}\\), where each heavier s-top (mass 582 GeV) decays into a lighter s-top \\(t_1\\) or \\(\bar{t_1}\\), mass 400 GeV) and a Higgs boson. The FCNC process involves \\(t\\bar{t}\\) production where one top quark decays to a Higgs boson and a light quark. We generate 10 million events for each process, except for \\(tHq\\) and \\(t\\bar{t}t\\bar{t}\\), where 5 million events were produced.

In all simulation samples, the center of mass energy of the proton-proton collision is set to 13 TeV. The Higgs boson, top quarks, and vector bosons are set to decay inclusively (except the \\(t\\bar{t}H \\rightarrow \\gamma\\gamma\\) samples), with MadSpin [Artoisenet et al., 2012](#ref-artoisinet-2012st) handling the decays of top quarks and W bosons. The generated events are processed through Pythia 8.235 [Sjostrand et al., 2015](#ref-sjostrand-2015) for parton showering and heavy particle decays, followed by Delphes 3.4.2 [de Favereau et al., 2014](#ref-defavereau-2014) configured to emulate the ATLAS detector [ATLAS Collaboration, 2008](#ref-atlas-2008) for fast detector simulation.

The detector-level object selection criteria are defined to align with typical experimental conditions. Photons are required to have transverse momentum \\(p_T \\geq 20~\\mathrm{GeV}\\) and pseudorapidity \\(|\\eta| \\leq 2.37\\), excluding the electromagnetic calorimeter crack region \\(1.37 < |\\eta| < 1.52\\). Electrons must have \\(p_T \\geq 10~\\mathrm{GeV}\\) and \\(|\\eta| \\leq 2.47\\) (excluding the same crack region), while muons are selected with \\(p_T \\geq 10~\\mathrm{GeV}\\) and \\(|\\eta| \\leq 2.7\\). Jets are reconstructed using the anti-\\(k_t\\) algorithm [Cacciari et al., 2008](#ref-cacciari-2008gp) with radius parameter \\(\\Delta R=0.4\\), where \\(\\Delta R\\) is defined as \\(\\sqrt{\\Delta\\eta ^2 + \\Delta\\phi^2}\\), with \\(\\Delta\\eta\\) being the difference in pseudorapidity and \\(\\Delta\\phi\\) the difference in azimuthal angle. Jets must satisfy \\(p_T \\geq 25~\\mathrm{GeV}\\) and \\(|\\eta| \\leq 2.5\\). To avoid double-counting, jets are removed if they are within \\(\\Delta R < 0.4\\) of a photon or lepton. The identification of jets originating from b-quark decays (b-tagging) is performed by matching jets within \\(\\Delta R = 0.4\\) of a b-quark, with efficiency corrections applied to match the performance of the ATLAS experiment's b-tagging algorithm [ATLAS Collaboration, 2019](#ref-atlas-2019bwq).

## Methods

### Overview

We present a methodology for developing and evaluating a foundation model for particle collision event analysis. The approach centers on pretraining a Graph Neural Network (GNN) architecture using a comprehensive dataset that spans multiple physics tasks, enabling the model to learn robust and transferable features. For task-specific applications, we employ a fine-tuning strategy that combines output layer adaptation with carefully calibrated learning rates for updating the pretrained parameters.

Given the prevalence of classification problems in particle physics data analysis, we evaluate the model's efficacy through a systematic assessment across five binary classification tasks:

- $t\bar{t}H(\rightarrow \gamma\gamma)$ with CP-even versus CP-odd t-H interaction
- $t\bar{t}$ with FCNC top quark decays versus $tHq$ processes
- $t\bar{t}W$ versus $ttt$ processes
- Stop pair production with Higgs bosons in the decay chain versus $t\bar{t}H$ processes
- $WH$ versus $ZH$ production modes

Our evaluation metrics encompass classification performance, computational efficiency, and model interpretability. The investigation extends to analyzing the model's scaling behavior with respect to training dataset size, benchmarked against models trained without pretraining. Although we explored transfer learning through parameter freezing of pretrained layers, this approach did not yield performance improvements, leading us to focus our detailed analysis on fine-tuning strategies.

This methodological framework demonstrates the potential of foundation models to enhance the efficiency of particle physics analyses while improving task-specific performance, offering a promising direction for future high-energy physics research.

---

### GNN Architecture

We implement a Graph Neural Network (GNN) architecture that naturally accommodates the point-cloud structure of particle physics data, employing the DGL framework with a PyTorch backend [Wang et al., 2019][ref-dgl-2019], [Paszke et al., 2019][ref-pytorch-2019]. A fully connected graph is constructed for each event, with nodes corresponding to reconstructed jets, electrons, muons, photons, and $\vec{E}_T^{\text{miss}}$. The features of each node include the four-momentum $(p_T, \eta, \phi, E)$ of the object with a massless assumption ($E = p_T \cosh \eta$), the b-tagging label (for jets), the charge (for leptons), and an integer labeling the type of object represented by the node. We use a placeholder value of 0 for features which are not defined for every node type such as the b-jet tag, lepton charge, or the pseudorapidity of $\vec{E}_T^{\text{miss}}$. We assign the angular distances ($\Delta \eta, \Delta \phi, \Delta R$) as edge features and the number of nodes $N$ in the graph as a global feature. We denote the node features $\{\vec x_i\}$, edge features $\{\vec y_{ij}\}$, and global features $\{\vec z\}$.

The GNN model is based on the graph network architecture described in [Battaglia et al., 2018][ref-graphnets-2018] using simple multilayer perceptron (MLP) feature functions and summation aggregation. The model is comprised of three primary components: an encoder, the graph network, and a decoder. In the encoder, three MLPs embed the nodes, edges, and global features into a latent space of dimension 64. The graph network block, which is designed to facilitate message passing between different domains of the graph, performs an edge update $f_e$, followed by a node update $f_n$, and finally a global update $f_g$, all defined below. The inputs to each update MLP are concatenated.

$$
\vec {y'}_{ij} = f_e\left(\{\vec x_k\},\vec y_{ij},\vec z\right) = \mathrm{MLP}\left(\vec x_i,\vec x_j,\vec y_{ij},\vec z\right)
$$

$$
\vec{x'}_{i} = f_n\left(\vec x_i,\{\vec{y'}_{jk}\},\vec z\right) = \mathrm{MLP}\left(\vec x_i,\sum_j\vec{y'}_{ij},\vec z\right)
$$

$$
\vec{z'} = f_g\left(\{\vec{x'}_i\},\{\vec{y'}_{ij}\},\vec z\right) = \mathrm{MLP}\left(\sum_i\vec{x'}_i,\sum_{i,j}\vec{y'}_{ij},\vec z\right)
$$

This graph block is iterated four times with the same update MLPs. Finally, the global features are passed through a decoder MLP and a final layer linear to produce the desired model outputs. Each MLP consists of 4 linear layers, each with an output width of 64, with the `ReLU` activation function. The output of the MLP is then passed through a `LayerNorm` layer [Ba et al., 2016][ref-layernorm-2016]. The total number of trainable parameters in this model is about 400,000.

As a performance benchmark, a baseline GNN model is trained from scratch for each classification task. The initial learning rate is set to $10^{-4}$ with an exponential decay following $LR(x) = LR_{\text{initial}}\cdot(0.99)^x$, where $x$ represents the epoch number.

---

### Pretraining Strategy

We explore two complementary pretraining approaches to develop robust representations of collision events: (1) multi-class classification, which trains the model to distinguish between different physics processes, and (2) multi-label classification, which predicts the existence and kinematics of heavy particles with prompt decays. The pretraining dataset consists of approximately 120 million events, evenly distributed across 12 distinct physics processes, including all major Higgs boson production mechanisms and top quark processes as described in [Data Samples](#sec-data). This large-scale pretraining effort was conducted on the Perlmutter supercomputer at NERSC.

#### Multi-class Classification

For Monte Carlo simulated events, the underlying physics process that generated each event is known precisely, providing natural labels for supervised learning. However, the challenge lies in the complexity of collision events: different physics processes can produce similar kinematics and event topologies, particularly in certain regions of phase space. No single observable can unambiguously identify the underlying process. By training the model to distinguish between 12 different processes simultaneously, we challenge it to learn subtle differences in kinematics and topology that collectively characterize each process. The model is trained using categorical cross entropy as the loss function. The output layer of the multiclass classification model has 832 trainable parameters.

#### Multi-label Classification

This approach combines both classification and regression tasks to characterize collision events. For discrete properties like particle presence in specific kinematic regions, we employ classification labels with binary cross-entropy loss. For continuous quantities like particle multiplicities, we use regression labels with mean-squared error loss. This hybrid approach enables the model to learn both categorical and continuous aspects of the physics processes simultaneously.

We develop a comprehensive set of 41 labels that capture both particle multiplicities and kinematic properties. This approach increases prediction granularity and enhances model interpretability. By training the model to predict event kinematics rather than event identification, we create a task-independent framework that can potentially generalize better to novel scenarios not seen during pretraining.

The particle multiplicity labels count the number of Higgs bosons ($n_{\text{higgs}}$), top quarks ($n_{\text{tops}}$), vector bosons ($n_V$), $W$ bosons ($n_W$), and $Z$ bosons ($n_Z$). The kinematic labels characterize the transverse momentum ($p_T$), pseudorapidity ($\eta$), and azimuthal angle ($\phi$) of Higgs bosons and top quarks through binned classifications.

For Higgs bosons, $p_T$ is categorized into three ranges: (0, 30) GeV, (30, 200) GeV, and (200, $\infty$) GeV, with the upper range particularly sensitive to potential BSM effects. Similarly, both leading and subleading top quarks have $p_T$ classifications spanning (0, 30) GeV, (30, 300) GeV, and (300, $\infty$) GeV. When no particle exists within a specific $p_T$ range, the corresponding label is set to $[0, 0, 0]$. For all particles, $\eta$ measurements are divided into 4 bins with boundaries at $[-1.5, 0, 1.5]$, while $\phi$ measurements use 4 bins with boundaries at $[-\frac{\pi}{2}, 0, \frac{\pi}{2}]$. As with $p_T$, both $\eta$ and $\phi$ labels default to $[0, 0, 0, 0]$ in the absence of a particle. This comprehensive labeling schema enables fine-grained learning of kinematic distributions and particle multiplicities, essential for characterizing complex collision events.

The loss function combines individual losses from all 41 labels through weighted averaging. Binary cross-entropy is applied to classification labels, while mean-squared error is used for regression labels. The model generates predictions for all labels simultaneously, with individual losses calculated according to their respective types. The final loss is computed as an equally-weighted average across all labels, with weights set to 1 to ensure uniform contribution to the optimization process. The output layer of the multilabel model has 2,688 trainable parameters.

#### Pretraining

During pre-training, the initial learning rate is $10^{-4}$, and the learning rate decays by 1% each epoch following the power law function $LR(x) = 10^{-4}\cdot(0.99)^x$, where $x$ is the number of epochs. Both pre-trained models reach a plateau in loss by epoch 50, at which point the training is stopped.

---

### Fine-tuning Methodology

For downstream tasks, we adjust the model architecture for fine-tuning by replacing the original output layer (final linear layer) with a newly initialized linear layer while retaining the pre-trained weights for all other layers. This modification allows the model to specialize in the specific downstream task while leveraging the general features learned during pretraining.

The fine-tuning process begins with distinct learning rate setups for different parts of the model. The newly initialized linear layer is trained with an initial learning rate of $10^{-4}$, matching the rate used for models trained from scratch. Meanwhile, the pre-trained layers are fine-tuned more cautiously with a lower initial learning rate of $10^{-5}$. This approach ensures that the pre-trained layers adapt gradually without losing their general features, while the new layer learns effectively from scratch. Both learning rates decay over time following the same power law function, $LR(x) = LR_{initial} \cdot (0.99)^x$, to promote stable convergence as training progresses.

We also evaluated a transfer learning setup in which either the decoder MLP or the final linear layer was replaced with a newly initialized component. During this process, all other model parameters remained frozen, leveraging the pre-trained features without further updating them. However, we did not observe performance improvements using the transfer learning setup. Consequently, we focus on reporting results obtained with the fine-tuning approach.

---

### Performance Evaluation

We assess model performance using two figures of merit: the classification accuracy and the Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve. The accuracy is defined as the fraction of correctly classified events when applying a threshold of 0.5 to the neural network output score. Both metrics demonstrate consistent trends in our analysis.

To obtain reliable performance estimates and uncertainties, we employ an ensemble training approach where 5 independent models are trained for each configuration with random weight initialization and random subsets of the training dataset. This enables us to evaluate both the models' sensitivity to initial parameters and to quantify uncertainties in their performance.

To investigate how model performance scales with training data, we conducted training runs using sample sizes ranging from $10^3$ to $10^7$ events per class ($10^3$, $10^4$, $10^5$, $10^6$, and $10^7$) for each model setup: the from-scratch baseline and models fine-tuned from multi-class or multi-label pretrained models. For the $10^7$ case, only the initialization was randomized due to dataset size limitations. All models were evaluated on the same testing dataset, consisting of 2 million events per class, which remained separate from the training process.

| **Name of Task**     | **Pretraining Task** | $10^3$         | $10^4$         | $10^5$         | $10^6$         | $10^7$         |
|----------------------|----------------------|----------------|----------------|----------------|----------------|----------------|
| **ttH CP Even vs Odd** | Baseline Accuracy     | 56.5 $\pm$ 1.1 | 62.2 $\pm$ 0.1 | 64.3 $\pm$ 0.0 | 65.7 $\pm$ 0.0 | 66.2 $\pm$ 0.0 |
|                        | Multiclass (%)        | +4.8 $\pm$ 1.1 | +3.4 $\pm$ 0.1 | +1.3 $\pm$ 0.0 | +0.2 $\pm$ 0.0 | -0.0 $\pm$ 0.0 |
|                        | Multilabel (%)        | +2.1 $\pm$ 1.2 | +1.9 $\pm$ 0.1 | +0.8 $\pm$ 0.1 | +0.0 $\pm$ 0.0 | -0.1 $\pm$ 0.0 |
| **FCNC vs tHq**        | Baseline Accuracy     | 63.6 $\pm$ 0.7 | 67.8 $\pm$ 0.4 | 68.4 $\pm$ 0.3 | 69.3 $\pm$ 0.3 | 67.9 $\pm$ 0.0 |
|                        | Multiclass (%)        | +5.8 $\pm$ 0.8 | +1.2 $\pm$ 0.4 | +1.4 $\pm$ 0.3 | +0.5 $\pm$ 0.3 | -0.0 $\pm$ 0.0 |
|                        | Multilabel (%)        | -5.3 $\pm$ 0.8 | -1.3 $\pm$ 0.4 | +0.9 $\pm$ 0.4 | +0.3 $\pm$ 0.3 | +0.4 $\pm$ 0.1 |
| **ttW vs ttt**         | Baseline Accuracy     | 75.8 $\pm$ 0.1 | 77.6 $\pm$ 0.1 | 78.9 $\pm$ 0.0 | 79.8 $\pm$ 0.0 | 80.3 $\pm$ 0.0 |
|                        | Multiclass (%)        | +3.7 $\pm$ 0.1 | +2.7 $\pm$ 0.1 | +1.3 $\pm$ 0.0 | +0.4 $\pm$ 0.0 | +0.0 $\pm$ 0.0 |
|                        | Multilabel (%)        | +2.2 $\pm$ 0.1 | +1.1 $\pm$ 0.1 | +0.5 $\pm$ 0.0 | +0.0 $\pm$ 0.0 | -0.1 $\pm$ 0.0 |
| **stop vs ttH**        | Baseline Accuracy     | 83.0 $\pm$ 0.2 | 86.3 $\pm$ 0.1 | 87.6 $\pm$ 0.0 | 88.5 $\pm$ 0.0 | 88.8 $\pm$ 0.0 |
|                        | Multiclass (%)        | +0.4 $\pm$ 0.2 | +1.9 $\pm$ 0.1 | +1.0 $\pm$ 0.0 | +0.3 $\pm$ 0.0 | +0.0 $\pm$ 0.0 |
|                        | Multilabel (%)        | +2.8 $\pm$ 0.2 | +1.0 $\pm$ 0.1 | +0.5 $\pm$ 0.0 | +0.0 $\pm$ 0.0 | -0.0 $\pm$ 0.0 |
| **WH vs ZH**           | Baseline Accuracy     | 51.4 $\pm$ 0.1 | 53.9 $\pm$ 0.1 | 55.8 $\pm$ 0.0 | 57.5 $\pm$ 0.0 | 58.0 $\pm$ 0.0 |
|                        | Multiclass (%)        | +5.2 $\pm$ 0.1 | +5.3 $\pm$ 0.1 | +3.1 $\pm$ 0.0 | +0.6 $\pm$ 0.0 | +0.1 $\pm$ 0.0 |
|                        | Multilabel (%)        | -1.1 $\pm$ 0.1 | -0.9 $\pm$ 0.2 | +0.5 $\pm$ 0.1 | +0.1 $\pm$ 0.0 | -0.1 $\pm$ 0.0 |

> **Table 1**: Accuracy of the traditional model versus the accuracy increase due to fine-tuning from various pretraining tasks.  
> The accuracies are averaged over 5 independently trained models with randomly initialized weights and trained on a random subset of the data. One exception is the $10^7$ training where all models use the same dataset due to limitations on our dataset size. The random subsets are allowed to overlap, but this overlap should be very minimal because all models take an independent random subset of $10^7$ events. The testing accuracy is calculated from the same testing set of 2 million events per class across all models for a specific training task. The errors are the propagated errors (root sum of squares) of the standard deviation of accuracies for each model.

## Results

### Classification Performance

Since the observations of AUC and accuracy show similar trends, we focus the presentation of the results using accuracy here for conciseness in Table 1.

In general, the fine-tuned pretrained model achieves at least the same level of classification performance as the baseline model. Notably, there are significant improvements, particularly when the sample size is small, ranging from \$10^3$ to \$10^4$ events. In some cases, the accuracy improvements exceed five percentage points, demonstrating that pretrained models provide a strong initial representation that compensates for limited data. The numerical values of the improvements in accuracy may not fully capture the impact on the sensitivity of the measurements for which the neural network classifier is used, and the final sensitivity improvement is likely to be greater.

As the training sample size grows to \$10^5$, \$10^6$, and eventually \$10^7$ events, the added benefit of pretraining diminishes. With abundant data, models trained from scratch approach or even match the accuracy of fine-tuned pretrained models. This suggests that large datasets enable effective learning from scratch, rendering the advantage of pretraining negligible in such scenarios.

Although both pretraining approaches offer benefits, multiclass pretraining tends to provide more consistent improvements across tasks, especially in the low-data regime. In contrast, multilabel pretraining can sometimes lead to neutral or even slightly negative effects for certain tasks and data sizes. This highlights the importance of the pretraining task design, as the similarity between pretraining and fine-tuning tasks in the multiclass approach appears to yield better-aligned representations.

Finally, the spread of accuracy across the five tasks for the baseline model is quite large, offering a robust test of fine-tuning across tasks of varying difficulty. The consistent observation of these trends across tasks confirms the reliability and robustness of the findings.

---

### Model Interpretability

We aim to understand whether pretrained and baseline models learn the same underlying representations. If the two models exhibit high similarity, a plausible interpretation is that pretraining provides the pretrained model with an advantageous initialization, allowing it to converge to a similar state as the baseline model more efficiently. Conversely, significant differences between the models would indicate that pretraining facilitates the development of a more general and robust latent space, which serves as a foundation for fine-tuning to effectively adapt to the downstream task. To investigate this, we analyzed the representational similarity between a pretrained model fine-tuned for the downstream task and a baseline model trained directly on the downstream task without pretraining.

We use Centered Kernel Alignment (CKA) [Kornblith et al., 2019][ref-kornblith-2019-cka] to analyze model similarity and interpretability. CKA is a robust metric that quantifies the similarity between the internal representations of neural networks by comparing their feature matrices in a manner that is invariant to scaling, rotation, and alignment. This invariance makes CKA particularly effective for studying relationships between network layers, even across networks of different sizes or those trained from varying initializations.

The similarity is evaluated using a 64-dimensional latent representation after the decoder stage of the GNN model. This choice allows us to compare the internal states of the models at a fine-grained level and understand how training strategies impact the representations directly used for the output task.

To provide an intuitive understanding of CKA values, we construct a table of the CKA scores for various transformations performed on a set of dummy data.

- **A:** randomly initialized matrix with shape (1000, 64), following a normal distribution ($\sigma = 1, \mu=0$)
- **B:** matrix with shape (1000, 64) constructed via various transformations performed on $A$
- **Noise:** randomly initialized noise matrix with shape (1000, 64), following a normal distribution ($\sigma = 1, \mu=0$)

| Dataset | CKA Score |
|---------|-----------|
| $A, B = A$ | 1.00 |
| $A, B =$ permutation on columns of $A$ | 1.00 |
| $A, B = A + \mathrm{Noise}(0.1)$ | 0.99 |
| $A, B = A + \mathrm{Noise}(0.5)$ | 0.80 |
| $A, B = A + \mathrm{Noise}(0.75)$ | 0.77 |
| $A, B = A\cdot \mathrm{Noise}(1)$ (Linear Transformation) | 0.76 |
| $A, B = A + \mathrm{Noise}(1)$ | 0.69 |
| $A, B = A + \mathrm{Noise}(2)$ | 0.51 |
| $A, B = A + \mathrm{Noise}(5)$ | 0.39 |

**Table 2:** CKA scores for a dummy dataset $A$ and $B$, where $B$ is created via various transformations performed on $A$.

As seen in Table 2 and in the definition of the CKA, the CKA score is permutation-invariant. We will use the CKA score to evaluate the similarity between various models and gain insight into the learned representation of detector events in each model (i.e., the information that each model learns).

We train ensembles of models for each training task to observe how the CKA score changes due to the random initialization of our models. The CKA score between two models is then defined to be:

$$
CKA(A, B) = \frac{1}{n^2}\sum_i^n \sum_j^n CKA(A_i, B_j)
$$

where $A_i$ is the representation learned by the $i^{th}$ model in an ensemble with $n$ total models. The error in CKA is the standard deviation of $CKA(A_i, B_j)$.

Here we present results for the CKA similarity between the final model in each setup with the final model in the baseline, shown in Table 3.

| Training Task         | Baseline         | Multiclass      | Multilabel      |
|----------------------|------------------|-----------------|-----------------|
| ttH CP Even vs Odd   | 0.94 ± 0.05      | 0.82 ± 0.01     | 0.77 ± 0.06     |
| FCNC vs tHq          | 0.96 ± 0.03      | 0.76 ± 0.01     | 0.81 ± 0.01     |
| ttW vs ttt           | 0.91 ± 0.08      | 0.75 ± 0.10     | 0.72 ± 0.05     |
| stop vs ttH          | 0.87 ± 0.11      | 0.79 ± 0.12     | 0.71 ± 0.08     |
| WH vs ZH             | 0.90 ± 0.07      | 0.53 ± 0.03     | 0.44 ± 0.06     |

**Table 3:** CKA Similarity of the latent representation before the decoder with the baseline model, averaged over 3 models per training setup, and all models trained with the full dataset (\$10^7$). The baseline column is not guaranteed to be 1.0 because of the random initialization of the model. Each baseline model converges to a slightly different representation as seen in the CKA values in that column.

The baseline models with different initializations exhibit high similarity values, ranging from approximately 0.87 to 0.96, which indicates that independently trained baseline models tend to converge on similar internal representations despite random initialization. Across the considered tasks, models trained as multi-class or multi-label classifiers exhibit noticeably lower CKA similarity scores when compared to the baseline model. For example, in the WH vs ZH task, the baseline model and another baseline trained model have a high similarity of 0.90, whereas the multi-class and multi-label models show significantly reduced similarities (0.53 and 0.44, respectively). This pattern suggests that the representational spaces developed by multi-class or multi-label models differ substantially from those learned by the baseline model that was trained directly on the downstream classification task.

### Computational Efficiency

To estimate the computational resources required for each approach, we measured the wall time needed for a model to reach its final performance. For baseline models, this is defined as the wall time from the start of training until the loss of the model plateaus. For the foundation model approach, the estimate includes both the pretraining time and the fine-tuning time, each measured from the start of training until the loss plateaus. This approach ensures a consistent and comprehensive evaluation of the computational demands.

![The ratio of the fine-tuning time required to achieve 99% of the baseline model's final classification accuracy to the total time spent training the baseline model.](training_time.png)
*Fig. 1: The ratio of the fine-tuning time required to achieve 99% of the baseline model's final classification accuracy to the total time spent training the baseline model.*

Figure 1 shows the fine-tuning time for the model pretrained with multiclass classification, relative to the time required for the baseline model, as a function of training sample size. In general, the fine-tuning time is significantly shorter than the training time required by the baseline model approach. For smaller training sets, on the order of $10^5$ events, tasks such as FCNC vs. tHq and ttW vs. ttt benefit substantially from the pretrained model’s “head start,” achieving their final performance in only about 1% of the baseline time. For large training datasets, the fine-tuning time relative to the baseline training time becomes larger; however, given that the large training sample typically requires longer training time, fine-tuning still yields much faster training convergence. The ttH CP-even vs. ttH CP-odd task, with a training sample size of $10^7$ events, is an exception where the fine-tuning time exceeds the training time required for the baseline model. This is likely because the processes involved in this task include photon objects in the final states, which are absent from the events used during pretraining.

To accurately evaluate the total time consumption, it is necessary to include the pretraining time required for the foundation model approach. The pretraining times are as follows:

- **Multi-class pretraining:** 45.5 GPU hours
- **Multi-label pretraining:** 60.0 GPU hours

The GPU hours recorded for the multi-label model represent the total time required when training the model in parallel on 16 GPUs. This includes a model synchronization step, which results in higher GPU hours compared to the multi-class pretraining model.

The foundation model approach becomes increasingly efficient when a large number of tasks are fine-tuned using the same pretrained model, compared to training each task independently from scratch. To illustrate this, we evaluate the computational time required for a scenario where the training sample contains $10^7$ events. For the five tasks tested in this study, the baseline training time (training from scratch) ranges from 1.68 GPU hours (WH vs. ZH) to 5.30 GPU hours (ttW vs. ttt), with an average baseline training time of 2.94 GPU hours. In contrast, the average fine-tuning time for the foundation model approach, relative to the baseline, is 38% of the baseline training time for $10^7$ events. Based on these averages, we estimate that the foundation model approach becomes more computationally efficient than the baseline approach when fine-tuning is performed for more than 41 tasks.

As a practical example, the ATLAS measurement of Higgs boson couplings using the $H \rightarrow \gamma\gamma$ decay channel [ATLAS Collaboration, 2023][ref-atlas-2023-higg] involved training 42 classifiers for event categorization. This coincides with our estimate, suggesting that the foundation model approach can reduce computational costs even for a single high-energy physics measurement.

## Conclusions

We presented an in-depth study of a particle physics foundation model designed to operate on the four-momentum and identification properties of event final-state objects. This model is built on a Graph Neural Network (GNN) architecture and trained on a dataset comprising 120 million simulated proton-proton collision events across 12 distinct physics processes. The pretraining phase explored both multiclass and multilabel classification tasks, providing a robust foundation for downstream applications. Notably, the pretrained models demonstrated significant improvements in event classification performance when fine-tuned, particularly for tasks with limited training samples.

The foundation model approach also offers substantial computational advantages. By leveraging fine-tuning, this methodology reduces the computational resources required for large-scale applications across multiple tasks. Our estimates indicate that significant resource savings can be achieved even for single particle physics measurements, making this approach both scalable and efficient.

To better understand the learned representations of the pretrained model and guide future optimization efforts, we employed a representational similarity evaluation framework using Centered Kernel Alignment (CKA). This metric allowed us to investigate the source of the performance gains observed in the foundation model. Our analysis revealed notable differences in the learned representations between the fine-tuned pretrained model and a baseline model trained from scratch. In deep learning, it is well-established that multiple equally valid solutions can exist. Future studies are necessary to determine whether the low similarity in latent representations reflects complementary information uniquely captured by the foundation and baseline models, or if it can simply be attributed to connected local minima in the loss landscape.

## Acknowledgments

This work is supported by the U.S. National Science Foundation under the Award No. 2046280, and by U.S. Department of Energy, Office of Science under contract DE-AC02-05CH11231.

## References

- <span id="ref-openai-2024-gpt4"></span> **OpenAI et al.** GPT-4 Technical Report. arXiv:2303.08774 (2024). [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774)

- <span id="ref-yosinski-2014-transfer"></span> **Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson.** How transferable are features in deep neural networks? CoRR abs/1411.1792 (2014). [http://arxiv.org/abs/1411.1792](http://arxiv.org/abs/1411.1792)

- <span id="ref-rombach-2021-latentdiffusion"></span> **Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.** High-Resolution Image Synthesis with Latent Diffusion Models. CoRR abs/2112.10752 (2021). [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)

- <span id="ref-podell-2023-sdxl"></span> **Dustin Podell, Zion English, Kyle Lacey et al.** SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv:2307.01952 (2023). [https://arxiv.org/abs/2307.01952](https://arxiv.org/abs/2307.01952)

- <span id="ref-jumper-2021-alphafold"></span> **John Jumper, Richard Evans, Alexander Pritzel et al.** Highly accurate protein structure prediction with AlphaFold. Nature 596, 583-589 (2021). [https://doi.org/10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

- <span id="ref-devlin-2018-bert"></span> **Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR abs/1810.04805 (2018). [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805)

- <span id="ref-atlas-2023-higg"></span> **ATLAS Collaboration.** Measurement of the properties of Higgs boson production at \\(\sqrt{s} = 13\,\text{TeV}\\) in the \\(H \to \gamma\gamma\\) channel using \\(139\,\text{fb}^{-1}\\) of \\(pp\\) collision data with the ATLAS experiment. JHEP 07 (2023) 088. [arXiv:2207.00348](https://arxiv.org/abs/2207.00348), [https://doi.org/10.1007/JHEP07(2023)088](https://doi.org/10.1007/JHEP07(2023)088)

- <span id="ref-atlas-2023-4top"></span> **ATLAS Collaboration.** Observation of four-top-quark production in the multilepton final state with the ATLAS detector. Eur. Phys. J. C 83 (2023) 496. [arXiv:2303.15061](https://arxiv.org/abs/2303.15061), [https://doi.org/10.1140/epjc/s10052-023-11573-0](https://doi.org/10.1140/epjc/s10052-023-11573-0)

- <span id="ref-kornblith-2019-cka"></span> **Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton.** Similarity of Neural Network Representations Revisited. CoRR abs/1905.00414 (2019). [http://arxiv.org/abs/1905.00414](http://arxiv.org/abs/1905.00414)

---

<!-- Historical/General Physics foundational texts -->

- <span id="ref-birell-1982-qfields"></span> **N. D. Birell, P. C. W. Davies.** Quantum Fields in Curved Space. Cambridge Univ. Press (1982).

- <span id="ref-feynman-1954"></span> **R. P. Feynman.** Phys. Rev. 94, 262 (1954).

- <span id="ref-einstein-1935-epr"></span> **A. Einstein, Yu. Podolsky, N. Rosen.** Phys. Rev. 47, 777 (1935).

- <span id="ref-berman-1983-stability"></span> **G. P. Berman, Jr., F. M. Izrailev, Jr.** Stability of nonlinear modes. Physica D 88, 445 (1983).

- <span id="ref-davies-1988-trapped"></span> **E. B. Davies, L. Parns.** Trapped modes in acoustic waveguides. Q. J. Mech. Appl. Math. 51, 477–492 (1988).

- <span id="ref-witten-2001"></span> **Edward Witten.** hep-th/0106109 (2001). [https://arxiv.org/abs/hep-th/0106109](https://arxiv.org/abs/hep-th/0106109)

---

<!-- Particle physics/data science foundational models -->

- <span id="ref-beutler-1994-hem"></span> **E. Beutler.** Williams Hematology, 5th Edition, Chapter 7, pp. 654–662. McGraw-Hill, New York (1994).

- <span id="ref-knuth-1973-fa"></span> **Donald E. Knuth.** The Art of Computer Programming vol. 1: Fundamental Algorithms, 2nd Ed., Addison-Wesley (1973).

- <span id="ref-smith-2005-philos"></span> **J. S. Smith, G. W. Johnson.** Philos. Trans. R. Soc. London, Ser. B 777, 1395 (2005).

- <span id="ref-smith-2010-jap-unpub"></span> **W. J. Smith, T. J. Johnson, B. G. Miller.** Surface chemistry and preferential crystal orientation on a silicon surface. J. Appl. Phys. (unpublished, 2010).

- <span id="ref-smith-2010-jap-sub"></span> **V. K. Smith, K. Johnson, M. O. Klein.** Surface chemistry and preferential crystal orientation on a silicon surface. J. Appl. Phys. (submitted, 2010).

- <span id="ref-underwood-1988-lowerbounds"></span> **Ulrich Underwood, Ned Net, Paul Pot.** Lower Bounds for Wishful Research Results. Talk at Fanstord University (1988).

- <span id="ref-johnson-2007-comm"></span> **M. P. Johnson, K. L. Miller, K. Smith.** Personal communication (Jan-May 2007).

---

<!-- Prototypical collider software and tools -->

- <span id="ref-pytorch-2019"></span> **Adam Paszke et al.** PyTorch: An Imperative Style, High-Performance Deep Learning Library. arXiv:1912.01703 (2019). [http://arxiv.org/abs/1912.01703](http://arxiv.org/abs/1912.01703)

- <span id="ref-dgl-2019"></span> **Minjie Wang et al.** Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs. arXiv:1909.01315 (2019). [http://arxiv.org/abs/1909.01315](http://arxiv.org/abs/1909.01315)

- <span id="ref-graphnets-2018"></span> **Peter W. Battaglia et al.** Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261 (2018). [http://arxiv.org/abs/1806.01261](http://arxiv.org/abs/1806.01261)

- <span id="ref-layernorm-2016"></span> **Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton.** Layer Normalization. arXiv:1607.06450 (2016). [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)

---

<!-- Recent & foundation models in HEP ML -->

- <span id="ref-wildridge-2024-bumblebee"></span> **Andrew J. Wildridge et al.** Bumblebee: Foundation Model for Particle Physics Discovery. arXiv:2412.07867 (2024). [https://arxiv.org/abs/2412.07867](https://arxiv.org/abs/2412.07867)

- <span id="ref-katel-2024-jet"></span> **Subash Katel et al.** Learning Symmetry-Independent Jet Representations via Jet-Based Joint Embedding Predictive Architecture. arXiv:2412.05333 (2024). [https://arxiv.org/abs/2412.05333](https://arxiv.org/abs/2412.05333)

- <span id="ref-araz-2024-pointcloud"></span> **Jack Y. Araz et al.** Point cloud-based diffusion models for the Electron-Ion Collider. arXiv:2410.22421 (2024). [https://arxiv.org/abs/2410.22421](https://arxiv.org/abs/2410.22421)

- <span id="ref-leigh-2024-maskedparticle"></span> **Matthew Leigh et al.** Is Tokenization Needed for Masked Particle Modelling? arXiv:2409.12589 (2024). [https://arxiv.org/abs/2409.12589](https://arxiv.org/abs/2409.12589)

- <span id="ref-mikuni-2024-omnilearn"></span> **Vinicius Mikuni, Benjamin Nachman.** OmniLearn: A Method to Simultaneously Facilitate All Jet Physics Tasks. arXiv:2404.16091 (2024). [https://arxiv.org/abs/2404.16091](https://arxiv.org/abs/2404.16091)

- <span id="ref-zhang-2024-xiwu"></span> **Zhengde Zhang et al.** Xiwu: A Basis Flexible and Learnable LLM for High Energy Physics. arXiv:2404.08001 (2024). [https://arxiv.org/abs/2404.08001](https://arxiv.org/abs/2404.08001)

- <span id="ref-harris-2024-resimulation"></span> **Philip Harris et al.** Re-Simulation-based Self-Supervised Learning for Pre-Training Foundation Models. arXiv:2403.07066 (2024). [https://arxiv.org/abs/2403.07066](https://arxiv.org/abs/2403.07066)

- <span id="ref-birk-2024-omnijet"></span> **Joschka Birk, Anna Hallin, Gregor Kasieczka.** OmniJet-$\alpha$: the first cross-task foundation model for particle physics. Machine Learning: Science and Technology. 5(3), 035031 (Aug 2024). [https://doi.org/10.1088/2632-2153/ad66ad](https://doi.org/10.1088/2632-2153/ad66ad)

- <span id="ref-huang-2024-lmtracking"></span> **Andris Huang et al.** A Language Model for Particle Tracking. arXiv:2402.10239 (2024). [https://arxiv.org/abs/2402.10239](https://arxiv.org/abs/2402.10239)

- <span id="ref-golling-2024-maskedset"></span> **Tobias Golling et al.** Masked Particle Modeling on Sets: Towards Self-Supervised High Energy Physics Foundation Models. arXiv:2401.13537 (2024). [https://arxiv.org/abs/2401.13537](https://arxiv.org/abs/2401.13537)

- <span id="ref-liu-2023-gaam"></span> **Junze Liu et al.** Generalizing to new geometries with Geometry-Aware Autoregressive Models (GAAMs) for fast calorimeter simulation. Journal of Instrumentation 18(11), P11003 (Nov 2023). [https://doi.org/10.1088/1748-0221/18/11/p11003](https://doi.org/10.1088/1748-0221/18/11/p11003)

- <span id="ref-hashemi-2024-gen"></span> **Baran Hashemi et al.** Ultra-high-granularity detector simulation with intra-event aware generative adversarial network and self-supervised relational reasoning. Nature Communications 15(1) (June 2024). [https://doi.org/10.1038/s41467-024-49104-4](https://doi.org/10.1038/s41467-024-49104-4)

- <span id="ref-vigl-2024-finetune"></span> **Matthias Vigl et al.** Finetuning Foundation Models for Joint Analysis Optimization. arXiv:2401.13536 (2024). [https://arxiv.org/abs/2401.13536](https://arxiv.org/abs/2401.13536)

- <span id="ref-li-2024-refine"></span> **Chen Li, Hao Cai, Xianyang Jiang.** Refine neutrino events reconstruction with BEiT-3. Journal of Instrumentation 19(6), T06003 (Jun 2024). [https://doi.org/10.1088/1748-0221/19/06/t06003](https://doi.org/10.1088/1748-0221/19/06/t06003)