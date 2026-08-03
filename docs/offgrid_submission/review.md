Paper Decision
Decisionby Program Chairs31 Jul 2026, 02:19 (modified: 31 Jul 2026, 06:00)EveryoneRevisions
Decision: Accept (Oral)
Meta Review of Submission30 by Area Chair 7kw6
Meta Reviewby Area Chair 7kw622 Jul 2026, 00:41 (modified: 31 Jul 2026, 05:58)EveryoneRevisions
Metareview:
This paper conducts a validation study to determine whether the reported gains of INR-based continuous registration methods stem from the neural network itself or from the deformation prior to it induces, comparing INR-Dense, INR-BSCP/SINR, D-BSCP, and MR-D-BSCP on brain MR (OASIS) and lung CT (DIR-LAB) registration across the alignment-regularity spectrum. All three reviewers rated the paper positively, agreeing it addresses an important, under-examined question with a rigorous evaluation protocol, particularly the missing B-Spline-only ablation for SINR and the use of trade-off curves instead of single-point comparisons. Beyond this shared praise, reviewers mostly raised distinct concerns rather than converging on one critique. BjCV finds the novelty limited, since D-BSCP/MR-D-BSCP amount to classical FFD with a dyadic pyramid, and separately argues that several claims (the OASIS spline-prior comparison, the DIR-LAB TRE parity with pTVReg, the "smooth/coherent" characterization of INR-Dense's prior) are stated qualitatively rather than statistically supported, with the regularizer left as an unexplored confound. n3j8 focuses on inconsistent baseline coverage across datasets and a narrow, seemingly arbitrary subset of cases in the supervised fitting experiment. NNPS and N3J8 both note that the ablation is incomplete without a multiresolution INR baseline, the one point of real overlap across reviews. Given unanimous acceptance-level ratings with reasonably high confidence and no disqualifying flaws, the recommendation is Accept.

Recommendation: Accept (Oral)
Confidence: 4: The area chair is confident but not absolutely certain
Interesting study but limited novelty
Official Reviewby Reviewer BjCV21 Jul 2026, 04:13 (modified: 31 Jul 2026, 05:58)EveryoneRevisions
Summary:
The paper presents a validation study of continuous deformable registration parameterizations, asking whether the reported gains of INR-based methods come from the neural representation itself or from the deformation prior it induces. The authors compare four models under a unified notation (INR-Dense/IDIR, INR-BSCP/SINR, direct single-scale B-Spline control-point optimization or D-BSCP, and a multiresolution variant MR-D-BSCP) on inter-subject brain MR (OASIS) and exhale-to-inhale lung CT (DIR-LAB), evaluated across the alignment-regularity spectrum rather than at a single operating point. They report that D-BSCP matches INR-BSCP on brain MR, that single-scale spline methods struggle under large respiratory motion while INR-Dense and MR-D-BSCP do not, and that MR-D-BSCP is the strongest of the tested parameterizations overall. A supervised target-field fitting diagnostic is added to separate representation capacity from optimization behaviour.

Strengths:
The question the paper asks very interesting, in my view, under-asked in the current INR registration literature. Adding a plain direct B-Spline optimization as a control for SINR is the ablation that was missing, and it is well executed. I also appreciate the evaluation protocol, since comparing across the full alignment-regularity trade-off rather than at a single selected operating point is a more careful choice than is common in this space, and it makes the OASIS comparison credible. The target-field fitting diagnostic is the part I found most interesting, because it supports the sharper and less obvious observation that the failure of single-scale methods under large deformation may be an optimization issue rather than a capacity one. The paper is also candid about its own loose ends, including the INR-Dense behaviour that fits poorly yet registers well, and the omission of Dual-INR.

Weaknesses:
D-BSCP and MR-D-BSCP are essentially classical FFD with a dyadic pyramid, and the benefit of coarse-to-fine schemes for large respiratory motion has been known for a long time, so the new contribution is the narrower SINR-specific one.

The statement that the spline prior accounts for much of SINR's behaviour rests on a comparison where the two methods differ not only in the INR but also in learning rate and implementation, and the OASIS difference is reported qualitatively rather than with numbers or a test. Similarly, 0.922 ± 0.152 against pTVReg's 0.919 ± 0.152 over ten cases is compatible with the two being indistinguishable, and the meaning of "matched to the pTVReg loss" is left undefined even though it bears directly on how much of that result comes from the parameterization. In the same spirit, the smooth and coherent prior attributed to INR-Dense is asserted rather than measured, which is worth noting given how much SIREN behaviour depends on ω₀.

The role of regularization is not explored. The paper frames the comparison largely in terms of parameterization and optimization path, but the regularizer is itself a strong prior, and a method that appears unsuited to a given task may simply be regularized in a way that does not reflect the specificities of that deformation, whether the displacements are large or small, smooth and coherent or locally varying. If, as it appears, the same regularization form was used throughout, it is hard to tell whether single-scale B-Spline methods are intrinsically ill-matched to DIR-LAB or whether an appropriately chosen regularizer would substantially narrow the gap. Varying the regularization form, and not only its strength, would speak more directly to the paper's own question.

Clarity And Organization: Good
Additional Comments:
Reporting runtime would strengthen the practical argument, as the speed advantage of direct optimization over per-pair MLP training seems to be a point in the authors' favour that is currently left unused.

Rating: 4: Weak Accept
Confidence: 3: Confident but not absolutely certain
Llm Acknowledgment: Yes
Best Paper Award: No
Justification For Rating:
Overall I find this paper borderline, and I lean towards weak accept. My reservations concern the limited novelty and a small number of claims that go further than the evidence allows. Set against that, the question is a very interesting one and the work is useful and nicely presented, so I would not object to acceptance.

Useful analysis of performance-driving components of "off-grid" registration methods (relevant for any future method development in that area)
Official Reviewby Reviewer nnps20 Jul 2026, 10:13 (modified: 31 Jul 2026, 05:58)EveryoneRevisions
Summary:
This paper conducts a validation study on continous registration methods and how their design (parametrization and optimization strategy) might affect registration performance and general capability. A dense INR approach is compared with B-spline parametrized versions (using INRs and single/multi-resolution direct optimization). Evaluation of registration performance as well as target field fitting (showing the overall potential independent of optimization) on two separate datasets show performance dependent of motion types.

Strengths:
This paper conducts a valid, comprehensive and structured comparison of off-grid regularization approaches. It is a relevant contribution to the field of continous registration, as it raises awareness to the importance of detailed component analysis of previous and novel methods and avoiding wrong claims to why a model might improve. The evaluation strategy gives a valid example on how the performance improvements can be evaluated. Hyperparameter selections are explained well and evaluation strategies seem comprehensive.

Weaknesses:
While the general idea of comparing conventional optimization vs. INR based B-Spline registration is valid and evaluated well, the results state that the multi-resolution version of conventional B-splines (MR-D-BSCP >> D-INR-BSCP) performs best or gives the best prior for deformations/optimization. Since the authors also claim that the B-spline parametrization helps INR-based regularization (D-INR-BSCP >> Dense-INR), it would be important to see if the multi-resolution approach would help the off-grid INR method as well to avoid the shown optimization problems in Table 3. The authors mention a multi-resolution INR version as non-trivial and unpublished in Section 4.1., but nevertheless, inclusion of this study/comparison would make the comparison study more complete and generalize the claim of induced deformation priors beyond the B-Spline representation (i.e. does multi-resolution help, independent of D or INR optimization).

Minor: The abbreviations get confusing, e.g. in the abstract INR-BSCP and SINR are introduced, and then sometimes one is used and then the other. INR-BSCP seems more fluent and it would be helpful to stay consistent thoruhgout the paper.

Clarity And Organization: Good
Additional Comments:
Thinking about future work it would be interesting to see the SOTA deep learning registration methods for more then one SDlogJ in Fig.2 (to compare the behaviour vs. the continous methods).

Rating: 5: Accept
Confidence: 3: Confident but not absolutely certain
Llm Acknowledgment: Yes
Best Paper Award: No
Justification For Rating:
The present paper conducts a valid comparison study, raising important awareness to the need to evaluate registration model components for their general representation capability. While the study might be strenghtened with some additional comparisons, it will nevertheless provide value to the community in the current form, justifying its direct acceptance.

The paper offers an interesting message with minimal weaknesses.
Official Reviewby Reviewer n3j820 Jul 2026, 07:09 (modified: 31 Jul 2026, 05:58)EveryoneRevisions
Summary:
The paper argues that the type of deep learning architecture is not the main contributor to higher performance in the field of image registration. The paper instead argues that the smoothness prior imposed onto the registration method (eg. spline interpolation) and its corresponding hyper-parameters are a larger contributor to performance, and these priors are also dependent on the data they are applied to.

Strengths:
Arguments for the right priors. The paper boldly positions itself against the unfortunately common trend of registration research that blindly searches for 'higher-performing' architectures. The authors propose well-motivated arguments in favour of paying attention to what kind of data (and thus which types of deformations) a given task includes. They then go on to show how for different registration tasks (brain and lung), a decent choice on control point spacing is the defining factor on the quality of registration result, as opposed to the inclusion of a neural network which offers no benefit.

Weaknesses:
Inconsistency in baselines across unsupervised experiments. The authors show a diverse and thorough initial set of baselines and hyper-parameters in the initial OASIS experiment, but only a small subset of those same baselines in the DIR-LAB experiment. Why are the cohort-based baselines (VoxelMorph, etc) not explored for the DIR-lab experiment? What about the control-point spacing hyper-params?

Supervised experiment results only shown for specific subjects. Despite the argumentation on why a supervised task provides insights on the representation capacity of specific methods, the authors only show results for 1 OASIS subject and 2 DIR-LAB subjects. Why are only these specific subjects shown? Why aren't the results averages of testing sets investigated in the previous section (n=100 for OASIS and n=10 for DIR-LAB)? What caused the authors to chose these specific subjects? Why aren't the same control point spacing hyper-parameters shown for this section either?

Clarity And Organization: Good
Additional Comments:
For figure 2, while the choice in using ellipses is an elegant choice to represent the variance along the x and y axes, the ellipses end up very densely overlapped making it hard to tell specific point's spread. Something else that leaves me hungry as a reader is not seeing the covariances of the result's distribution to see what kind of correlation exists between SDlogJ and the segmentation metrics. While I wouldn't know how to more clearly design a visualization for such data, and I'd expect that resorting to using error bars along both x and y for all points would result in equally cluttered plots, I wonder whether the figures would benefit from some disentangling of the axis into individual 1-dimensonial violin plots. I would be curious to hear the design choices and findings that the authors went through when experimenting with data visualization.

What kind of layer depths where used in the INR-based baselines? How do the authors expect the depths to correlate with performance (both for the unsupervised as well as supervised cases)?

Is there a reason a multi-resolution spline approach was used in the direct case, but not for the INR spline baseline? How would the authors expect a multi-resolution INR-BSCP method to compare?

Formatting comments:

Currently, Figures 2 and 3 are positioned very far from where they are referenced in the text. I think it would help readers better follow the paper if in the camera-ready version they were positioned closer (or preferably in the same page) to where they are referenced in the text body.
Rating: 5: Accept
Confidence: 4: Very confident
Llm Acknowledgment: Yes
Best Paper Award: No
Justification For Rating:
The paper offers a very down-to-earth approach to image registration, showing evidence to the use of deep learning is often contributes little of not at all compared to having a deeper understanding of the registration problem at hand. I believe these types of works are critical to help dampen the current trends in the fields that regard the act of blindly throwing deep learning at a problem as a reasonable all-purpose solution to every problem. While the results could benefit from being a bit wider, more through, and better visualized, I believe this to be a minimal weakness to the paper. I believe the message of the paper thorough and well-described, as well as correctly explored by its experiments.