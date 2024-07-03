**This file is not mandatory**

In part 2 of Lab5, I had to use additional library pytorch-crf==0.7.2 
The reason is, the library is used in order to imporve slot filling prediction.
You may have get an error of "no module named 'torchcrf'" because of it.
However, simply adding pytorch-crf==0.7.2 to the requirements would solve the problem.

The the size of the model (bin file) is too big even though I included in the submission.

Also I was getting an error of "no module named 'transformers'" even I had installed
the requirements (when trying to run the code on my machine without a cuda)
if the same error appears, simply installig the library 
pip install transformers==4.38.0 would solve the problem.
