<div align="left">
  <h1>Basketball Shot Assistant</h1>
  <h3>Computes the optimal shooting motion parameters</h3>
</div>
<br/>
<br/>


## Collecting the necessary data ##
This application requires collecting data from a significant sample size of shots by a basketball player.

The application supports any two shooting motion parameters at a time. In addition, assign to each pair of parameters the success of the shot: miss, make or swish.

Examples of shooting motion parameters: body-arm angle, elbow angle, angle of knee bend, maximum height of the ball during shot, etc.

**IMPORTANT**: 
1) All measured parameters must be measured the same way on every shot/sample. Negative example: measuring the elbow angle at the set shooting position in one sample, and at the release point in another sample.

2) The shots in a dataset must be taken from the same spot on the basketball floor.

3) For the effectiveness of the program, the dataset must consist of at least around 50 shots. Having a larger dataset improves the prediction.
<br/>

## Storing the acquired data ##
The data must be stored in a text file, in CSV file format (without the header row). A row stands for the parameters of a shot: [parameter1],[parameter2],[shot result].

The shot result is a digit: 0 (miss), 1 (make) or 2 (swish).

Two sample datasets have been included in the extras folder.

Example of a row in the dataset:

114.84,126.44,1
<br/>

## Using the program and interpreting the results ##
The use of the program is intuitive and easy: select the text file with the dataset and wait for the results.

Once the calculations have been finished, the optimal parameters and the shot percentage at those parameters will be shown.

In addition, three plots will appear. The first one just plots the selected dataset. The second one shows the most effective range of parameters. The third one is a colormap of probabilities, showing the probability of making the shot at different parameters.
<br/>

## Final notes ##
My intent is to keep working on the application and expand to more than the two mentioned parameters.

A great way to support me would be to send your datasets via my public email or even in the issues tab.

Any help is greatly appreciated!
