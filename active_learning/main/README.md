# Active learning

## Overview

This project demonstrates how to train a predictor using both simulating the active learning process, and then leverage Active Learning strategies to optimize the best metal alloy ratios. The project includes the following main components:

1.  **(Predictor) Element Embeddings with OpenAI API & Predictive Modeling**

This repository showcases a practical application of generating element embeddings using the **OpenAI API**. These embeddings provide dense, context-rich numerical representations of chemical elements, which can be invaluable for various machine learning tasks in scientific domains.

To demonstrate the utility of such features, these embeddings were used in conjunction with a **Gradient Boosting Regressor**. Our predictive model achieved a notable **test correlation of 0.94**, highlighting the effectiveness of combining sophisticated feature engineering with robust machine learning algorithms. The code provides a clear example of the API integration for obtaining these element-specific embeddings.

2.  **(Simulated) Simulated Active Learning Demonstrations**:
    * **Knowledge-Augmented Bayesian Optimization (KABO)**: Showcases the application of KABO in a simulated environment.
    * **Bayesian Optimization with Policy Improvement Constraints (BOPIC)**: Demonstrates BOPIC within a simulated environment.
<img width="800" alt="portfolio_view" src="https://raw.githubusercontent.com/zhang21mit/CRESt/main/active_learning/main/Simulated/overall.png">
The figure above illustrates the comparative results of Knowledge-Augmented Bayesian Optimization (KABO) and Bayesian Optimization with Policy Improvement Constraints (BOPIC) in a simulated environment. To ensure a fair comparison, all experiments were conducted under the same standards: Bayesian optimization iterations (bo_iterations) were set to 20, and the batch size (bo_batch_size) was set to 10. Each configuration was run for 20 repetitions, and the results were then plotted to generate the figure shown. The detailed experimental result data has been stored in the result folder. 

3.  **(Experiments) Experimental data with CRESt for catalyst discovery**:
    * All experimental data related to this project could be found in Data_AL_1-3.
