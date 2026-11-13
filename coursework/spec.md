# CM32032: Reinforcement Learning — Assessment 1: Function-Approximated RL Project

> Text transcription of `CM32032-CW1-spec-26-27.pdf` (same folder), made 2026-11-13 so the spec can be read and searched as text. The PDF is the original: if the two ever differ, the PDF wins. Wording is verbatim; only layout was changed (line breaks, page numbers removed, link targets written out).

Joshua B. Evans, Sophia Jones

| Field | Value |
| --- | --- |
| **Date Set** | 2 November 2026 |
| **Date Due** | 7 January 2027, 20:00 |
| **Group Formation Deadline** | 6 November 2026, 20:00 |
| **Percentage of Overall Unit Mark** | 50% |
| **Submission Location** | Moodle |
| **Submission Components** | Project Report, Video Presentation, Agent Performance Video, Source Code, Group Contribution Form |
| **Submission Format** | `.pdf`, video files (e.g., `.mp4`), and `.zip` |
| **Anonymous Marking** | No |
| **Generative AI Assessment Lane** | Open Lane ([guidance](https://teachinghub.bath.ac.uk/generative-ai-and-education-at-the-university-of-bath/)) |

## What You Need To Do

Apply function-approximated reinforcement learning methods to a problem of your choice. You may choose any problem that engages your interest, but **its state/action space must be large enough that tabular reinforcement learning methods cannot be used to solve it effectively**.

Some problems that have been tackled in past projects include

- [teaching a simulated bipedal robot how to navigate an obstacle course](https://gymnasium.farama.org/environments/box2d/bipedal_walker/),
- [teaching a four-legged creature how to walk](https://gymnasium.farama.org/environments/mujoco/ant/),
- [driving a simulated car around a racetrack](https://gymnasium.farama.org/environments/box2d/car_racing/),
- [playing Atari arcade games](https://ale.farama.org/environments/),
- [playing Super Mario Bros.](https://github.com/Kautenja/gym-super-mario-bros),
- [landing a simulated spacecraft on the moon](https://gymnasium.farama.org/environments/box2d/lunar_lander/),
- [playing tile-matching games like Candy Crush and Bejeweled](https://github.com/akshilpatel/tile-match-gym), and
- [swinging up a simulated double pendulum](https://gymnasium.farama.org/environments/classic_control/acrobot/).

Your objective is to investigate and evaluate the effectiveness of reinforcement learning methods in solving your chosen problem, and to demonstrate a successful solution based on your findings. You may use any **function-approximated** reinforcement learning methods you believe to be appropriate, including methods we have not explicitly covered in the unit.

You must work in groups of four to six students. You will have the option of forming your own groups on Moodle; otherwise, you will be assigned to a group randomly. Multiple groups are allowed to work on the same problem.

## What You Need To Submit

For this project, you will need to submit five deliverables:

- a project report,
- a video presentation,
- a video of your agent(s) before and after learning,
- your source code, and
- a group contribution form.

You should upload your project report and group contribution form as two separate `.pdf` files; please do not zip/compress them. You should upload your videos as two separate files using a common format (e.g., `.mp4`, `.mov`, or `.mkv`). You should upload your source code compressed into a single `.zip` file.

If any of your submission files are too large to upload to Moodle directly, you may upload a text file (`.txt`) containing links to versions of them hosted online. In such cases, the links must remain publicly accessible for at least one year after the submission deadline. One member of each group should upload all five deliverables on behalf of the group.

### Report Structure

Your project report *must* be divided into the following sections, in the order given:

1. **Problem Definition:** A clear, precise, and concise description of your chosen problem, including its states, actions, transition dynamics, and reward function. You should discuss the difficulty of your chosen problem and justify why it cannot be solved effectively using tabular reinforcement learning methods.
2. **Background:** A discussion of reinforcement learning methods that may be effective at solving your chosen problem, including their relative strengths and weaknesses. You should consider both the approaches you intend to investigate and relevant alternatives, drawing on existing results from the scientific literature or other appropriate sources.
3. **Method:** A description of the methods you implemented and investigated, including any significant variants or refinements. You should explain how these methods work (in your own words), justify your methodological choices, and explain the reasoning behind any changes or refinements you made.
4. **Results:** A presentation and comparison of your experimental results, showing how quickly and how well your agent(s) learn (i.e., improve their policies). You should include informative baselines and meaningful comparisons between the approaches you investigated.
5. **Discussion:** A critical evaluation of your findings, including how effectively your approaches solved the chosen problem, their relative strengths and weaknesses, and what conclusions can be drawn from your experiments. You should also consider the limitations of your investigation.
6. **Future Work:** A discussion of potential future work you would pursue if you had more time, motivated by your findings and the limitations of your investigation.
7. **Personal Experience:** A brief discussion of your experience with the project, such as any unexpected difficulties, pleasant surprises, or lessons learned while completing it.
8. **Generative AI Use Declaration:** A brief statement indicating whether your group used GenAI tools and, if so, describing which tools were used, for what purposes, and to what extent. Where relevant, identify how different group members used these tools. This section does not count towards the report page limit.
9. **References:** A complete list of all sources cited in your report. You should cite relevant literature for all methods and ideas you discuss, regardless of whether you first encountered them in lectures or through your own reading. You may use any appropriate citation and referencing format, as long as it is applied consistently. References do not count towards the report page limit.
10. **Appendices:** Appendices may include (1) a detailed description of the problem domain; (2) experimental details sufficient to reproduce your experiments; (3) details of how you selected your hyperparameters, where applicable; and (4) additional supporting results beyond those presented in the main body of your report. Appendices should be clearly referenced in the main body of your report. Appendices do not count towards the page limit.

Your report should be written using the provided **LaTeX template**, and must not exceed **seven pages**, including figures, tables, and their captions, but excluding the Generative AI Use Declaration, References, and Appendices. The main body of your report should contain all information necessary to understand and evaluate your work, including your chosen methods, results, comparisons, and conclusions. Appendices should contain additional supporting material rather than essential content. The content of all figures, including any embedded text, should be clearly legible; you will lose marks for figures and text that are too small to view comfortably. You should not modify the provided LaTeX template. Due to the video presentation component, this coursework is not marked anonymously.

### Video Presentation

Your video presentation should summarise the most important information in your report. Somebody should be able to watch your video and leave with a good understanding of your chosen problem, the solution methods you investigated, and your main results.

Aside from that, you are free to organise your video presentation as you wish. Your video presentation should be no longer than **four minutes**, recorded at your natural speaking tempo. All group members must take part (i.e., speak) in the video presentation.

### Video of Agent Performance

You must submit a video showing your agent’s performance before and after training, clearly demonstrating that learning has occurred. This video should be at most **two minutes** long, but can be shorter as long as it demonstrates learning. You may speed up this video as long as the content remains clear.

### Source Code

You should submit all source code used to complete this project. We should be able to run your code and reproduce your results ourselves if needed, so you should include a short README file explaining how to do so.

### Group Contribution Form

You will be working as part of a group to complete this project. As such, you should ensure that you organise your time and resources effectively; finding time to plan, set expectations, discuss your work, and agree on each group member’s responsibilities is essential.

We have provided you with a group contribution form to fill out and submit. To fill out this form, you must agree on the relative contributions of each group member to your final project. We will use these relative contributions to scale marks where appropriate. For more information, please see the provided group contribution form document.

## Marking Criteria

Taking into account your report, presentation, agent demonstration video, and code, we will assess your work against the following four criteria:

- **Quality:** How effectively did you solve your chosen problem, taking into account its difficulty and the time and computational resources available to you? How appropriate, sophisticated, or creative were the methods you implemented?
- **Understanding:** How deep is your understanding of reinforcement learning in general, and of the solution methods you investigated in particular? Have you made principled decisions when selecting, developing, and refining your methods? Can you explain and critically evaluate the results of your investigation?
- **Rigour:** How thoroughly did you evaluate and compare the approaches you investigated? Was your experimental procedure sound? How informative were your investigations, and how convincingly do your results support your conclusions?
- **Presentation:** How clear, concise, well-organised, and engaging are your report, video presentation, and agent demonstration?

We are not interested solely in how well your final agent performs. Strong submissions will also demonstrate a deep understanding of the methods investigated, well-justified methodological decisions, rigorous experimentation, critical analysis of results, and clear communication of conclusions.

## Grade Descriptors

The following grade descriptors provide general guidance on the standard of work expected to reach a selection of grade boundaries. Please treat these as illustrative rather than a strict checklist of requirements.

**To achieve a grade above 50%**, students must implement an agent that demonstrates noticeable learning in their chosen domain. They may use relatively basic methods covered in lectures and conduct only a limited evaluation, with few or no comparisons against useful baselines or alternative approaches. Students should explain how their chosen methods work and provide some justification for their decisions, but may show limited awareness of relevant alternatives. Their analysis may simply describe the results, without explaining why they conducted particular experiments or what the results tell them about the effectiveness of their methods. The report and video presentation should cover the main aspects of the project, but may contain weaknesses in clarity, organisation, or correctness. Students may present figures without adequately explaining what they show or how they relate to their findings.

**To achieve a grade above 60%**, students should implement an agent that demonstrates substantial learning relative to the difficulty of their chosen problem and the resources available. They should select appropriate methods, explain how they work, and justify their main methodological decisions. Students should evaluate their agents using informative baselines and meaningful comparisons between the methods or variants they investigate. They should identify the main differences in their results, explain how these relate to their chosen approaches, and draw conclusions supported by experimental evidence. Students should demonstrate a good understanding of their chosen methods, including their strengths and weaknesses, and an appreciation of relevant alternatives. The report and video presentation should clearly explain the investigation and its findings, using well-presented figures and results to support the discussion.

**To achieve a grade above 70%**, students should implement an agent that achieves a high level of performance relative to the difficulty of their chosen problem and the resources available. Their work should go beyond a straightforward application of methods covered in lectures, whether through implementing more advanced methods, introducing well-justified algorithmic refinements, or performing particularly insightful experimentation and analysis. Students should design experiments to answer specific research questions, selecting baselines and comparisons that help them evaluate their approaches. They should investigate why their approaches performed differently, assess the reliability of their comparisons, and explain how any experimental limitations affect their conclusions. Students should demonstrate a deep understanding of their chosen methods and relevant alternatives, clearly justifying their methodological decisions and explaining their effects on learning. The report and video presentation should clearly connect the questions investigated, the experiments conducted, and the conclusions drawn, using carefully selected figures and results to support their arguments.

**This coursework has been designed such that the mean grade is expected to fall between 50% and 70%.**

**Grades above 70% will normally be rare.**

## Feedback & Support

We strongly encourage all groups to attend lab sessions regularly to discuss their chosen problem, solution methods, and project progress with tutors. This is an opportunity to receive **formative feedback**, identify potential issues early, and ensure your project is progressing in a sensible direction.

Each group will receive **summative feedback** on their submission within fifteen working days of the submission deadline, including an overall grade and feedback pertaining to each of the four marking criteria. Each student will receive an individual grade based on their group’s overall grade and adjusted according to their group contribution form. Additionally, class-wide feedback will be posted on the unit’s Moodle forums.

## Penalties

All five deliverables are mandatory: you must submit a project report, presentation video, agent demonstration video, source code, and a completed group contribution form. A penalty of **–10 marks** will be applied for each deliverable that is missing from your submission.

A penalty of **–5 marks** will also be applied for each of the following issues: exceeding the specified page or time limits, not using the provided LaTeX template to write the report, not following the prescribed report structure, not recording the video presentation at your normal speaking tempo, or not having all group members participate in the video presentation.

We will not read beyond the first **seven pages** of the main body of your report, and no marks will be given for content beyond this point. The same applies to your videos: we will not watch or award marks for content beyond the specified **time limits**.

Penalties will be applied **cumulatively**. If multiple issues apply to a submission, then the corresponding penalties will be **added together**. For example, a submission that exceeds the page limit and does not follow the prescribed structure will incur a total penalty of –10 marks.

There is no cap on the total penalties a submission may incur. However, penalties will not turn an otherwise passing mark into a fail. If a submission achieves a passing mark before penalties, its final mark will not be reduced below the pass threshold, regardless of the total penalties incurred.

Choosing a problem that can be solved effectively using tabular reinforcement learning methods will result in a **failing grade**. You are encouraged to get feedback on your chosen problem from tutors during lab sessions to ensure that its state/action space is large enough to be suitable for this project.

## Academic Integrity

You must develop your solution and write your report independently of other groups. Cite literature sources appropriately and acknowledge any third-party material or external assistance you use.

**Ultimately, you remain responsible for everything you submit**, including verifying its accuracy and ensuring your understanding of it. Each group member is expected to understand the overall approach, implementation, and results you present in your report, and to be able to explain their own contributions to the project overall.

### Using Generative AI

This is an **Open Lane** assessment. You may use generative artificial intelligence (“GenAI”) tools to support your work, but their use is not required.

GenAI tools *can* help with tasks such as exploring ideas, understanding technical material, and developing implementations. However, their outputs may be inaccurate, misleading, or inappropriate for your chosen problem. You should therefore exercise critical judgement and carefully verify any outputs you incorporate into your submission.

You must honestly and accurately acknowledge any use of GenAI tools in the Generative AI Use Declaration section of your report.

For further information about the University’s approach to GenAI in assessment, please see the [University’s guidance on GenAI in education and assessment.](https://teachinghub.bath.ac.uk/generative-ai-and-education-at-the-university-of-bath/)

### Using Third-Party Code

You may use third-party libraries to provide generic functionality or to implement your chosen environment. For example, you may use libraries like TensorFlow or PyTorch to implement neural networks, or use an existing environment implementation from a library like Gymnasium.

You must develop your own implementation of at least one function-approximated reinforcement learning method, rather than importing a ready-made implementation from an existing library or repository. You will not receive marks for methods you have not implemented yourselves. A submission that does not include such an implementation will receive a **failing grade**.

The *only* instance in which it is permissible to use an existing implementation of a reinforcement learning algorithm is if you include it solely as a baseline to compare the performance of your own implementation(s) against. If you choose to do this, you must make it clear which algorithms you have implemented yourself and which use existing implementations.

If you incorporate code from third-party sources into *any* part of your implementation, you must acknowledge this in your README file. For each source, provide a reference, explain how you used the code, and describe any modifications or extensions you made on top of it.

### Academic Misconduct

Do not plagiarise, fabricate results, or collude with other groups. Academic misconduct is a serious offence, and both your source code and report *will* be checked for evidence of it.

If concerns arise about the academic integrity of a submission, students may be invited to explain and defend their work through an individual oral discussion (a “Presentation of Learning”) as part of the University’s academic misconduct investigation procedures.

For further information, please see the [University’s guidance on academic misconduct.](https://www.bath.ac.uk/guides/academic-misconduct/)

## Frequently Asked Questions

### How many reinforcement learning methods should we compare?

There is no prescribed minimum number of methods beyond the requirement to implement at least one function-approximated reinforcement learning method yourself. You might compare different algorithms, investigate successive variants or refinements of a single algorithm, or evaluate combining ideas from various approaches.

### Why is there such a strict seven-page limit for the report?

The report template is based on the format used by *Neural Information Processing Systems*, a leading machine learning conference. As such, the page limit and format broadly reflect the constraints researchers face when publishing their work. Communicating technical material clearly and concisely is a valuable skill in both industry and academia.

References, appendices, and your GenAI use declaration do not count towards the seven-page limit. However, your main report must contain all the information necessary to understand and evaluate your work.

### Will the size of our group be taken into account when marking?

We do not apply a fixed adjustment to marks based on group size, but we recognise that larger groups may be able to undertake more work. We will consider this context when assessing your submission, although we do not expect substantial differences in scope between groups of four and six.

### Can I work individually or in a smaller group?

No. You must work in a group of four to six students, as specified in the coursework instructions.

Exceptions will only be considered where an approved Disability Access Plan (DAP) specifically identifies group work as an issue and suggests reasonable adjustments. If this applies to you, please contact the Unit Convenor and your Director of Studies before the group formation deadline.

### Is our chosen problem too simple?

We strongly encourage you to discuss your chosen problem with a tutor during a lab session before committing substantial time to it.

If your problem has a sufficiently small, discrete state/action space that it can be solved effectively using tabular methods such as Q-Learning, then it is not suitable for this coursework. Problems with continuous or very large state spaces are generally more appropriate.

### Is our chosen problem too ambitious?

We strongly encourage you to discuss your chosen problem with a tutor during a lab session before committing substantial time to it.

We also strongly recommend developing and testing your implementations on a simpler initial environment that still requires function approximation, before attempting your final, more complex target problem. This will help you identify implementation errors and verify that your methods work quickly, as well as give you some preliminary results to fall back on if you encounter difficulties with your more complex environment.

### Do we need to choose a problem that hasn’t been solved before?

No. You are welcome to choose a problem that has already been tackled in the research literature, by previous student groups, or by other groups in your cohort. You are not expected to solve a novel problem, but you must develop and evaluate your own solutions independently of other groups.

### Can we use a pre-existing environment or simulator?

Yes. In fact, we strongly recommend using an existing environment or simulator where possible. Developing an environment from scratch can be very time-consuming, and there are no marks available for doing so. You’re better off spending your effort developing, evaluating, and understanding your reinforcement learning agents.

### What should we use as baselines for comparison?

Choose baselines that help you evaluate your methods. Depending on your problem, these might include a random agent, a simpler reinforcement learning method, an earlier version of your implementation, human performance, or results reported in the literature. You should justify your choice of baselines and ensure that your comparisons are meaningful.

### Should we include all the methods and experiments we tried?

Your main report should focus on the methods and experiments most relevant to your investigation. You do not need to discuss every unsuccessful attempt, although negative results can be valuable when they help explain your findings or justify your decisions.

You should provide sufficient information to reproduce your reported experiments, including all relevant hyperparameters, experimental settings, and implementation details, as well as justifications for how you arrived at them. You may place supporting results and technical details in the appendices.

### Do our agents need to achieve state-of-the-art performance?

No. We assess the performance of your agents relative to (1) the difficulty of your chosen problem and (2) the time and computational resources available. We do not expect you to match or exceed the best results reported in the research literature.

### How much learning do our agents need to demonstrate?

At a minimum, you must demonstrate noticeable learning in at least one agent on a problem suitable for function-approximated reinforcement learning. Your agent should clearly outperform an agent that selects actions randomly, with evidence of learning both qualitatively (e.g., behaviour shown in your agent demonstration video) and quantitatively (e.g., learning curves presented in your report).

We do not expect every group to achieve optimal performance on their chosen problem. As the Quality criterion explains, we assess agent performance relative to (1) the difficulty of your chosen problem and (2) the time and computational resources available. Remember that agent performance is only one aspect of Quality, which is itself just one of the four marking criteria.

### What happens if our agent fails to learn?

As the grade descriptors explain, demonstrating noticeable learning on a suitable problem is a minimum expectation for achieving a passing grade. If none of your agents demonstrates learning on such a problem, you will not meet this requirement.

We recommend testing your implementations early, monitoring learning throughout development, and seeking advice from tutors when you encounter difficulties. If an approach fails, investigate the reasons and explain your findings in the report.
