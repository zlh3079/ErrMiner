# ErrMiner
A tool for mining error information from live chat data.
## 1 Overview of ErrMiner
Software developers frequently use live chat to ask questions, confirm or refute the existence of problems, and suggest solutions or recommendations. Therefore, these chats often contain a significant amount of error-prone information. However,due to the fleeting and intricate nature of live chat messages, labeling and recording such content require substantial time and labor resources, making it difficult to provide adequate support for existing methods that demand a large number of training samples. Furthermore, since live chat messages from various chat rooms often pertain to diverse problem domains, it is necessary to train a highly adaptable model.To address these challenges, we propose a method, called LRLC, that accumulates knowledge from live chat messages originating from distinct problem domains. The method learns to recognize complex context representations and distinguish error message dialog boxes, quickly adapting to new error mining tasks through further fine-tuning when encountering novel problem domains. LRLC consists of two parts: an error information mining component that extracts intricate context representations from live chat messages and identifies error message dialog boxes, and a transfer learning component that enhances the model's learning ability and cross-domain versatility.
Experimental results indicate that when applied to error mining tasks for live chat messages in entirely new problem domains, LRLC outperforms the best baseline method by 5.42%, 7.20%, and 6.14% in P (Precision), R (Recall), and F1-score (harmonic mean of Precision and Recall), respectively, achieving 83.24%, 85.23%, and 83.88%.
## 2 Project Structure
- `data/`
	- `*Processedn :he data set as processed and publicly released by Shi et al`
	- `*Augmented :the data set after augmentation and expansion`
	- `*Deduplicated :our data set after deduplication processing`


- `dataloader.py : dataset reader for ErrMiner`
- `model.py : ErrMiner model`
- `FocalLoss.py : focal loss function`
- `train.py : a file for model pre-training`
- `transfer.py : a file for model fine-tuning`
## 2 How it operates
Firstly, we pre-train on the Deduplicated dataset. Secondly, we fine-tune using the Augmented dataset. Lastly, we test the results on the Processed dataset.
