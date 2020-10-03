<div align="left">
  <h1>Basketball Shot Assistant</h1>
  <h3>Computes the optimal Shooting motion parameters</h3>
</div>
<br/>


## Collecting the necessary data ##
This application requires collecting data from a significant sample size of shots by a basketball player.

The supported parameters are the body-arm angle and the elbow angle, measured either at the set shooting position (right before starting to elevate for the shot) or at the moment of the release. In addition, assign to each pair of angles the success of the shot: miss, make or swish.

**IMPORTANT**: 
1) Choose at which moment to measure the angles from the start. All measured angles must belong to one of the two moments mentioned previously. Negative example: having half of the samples measured at the set shooting position and the other half at the moment of the release.
2) The shots in a sample must be taken from the same spot on the basketball floor.
3) For the effectiveness of the program, the dataset must consist of at least around 50 shots.


## Storing the acquired data ##
The data must be stored in a text file, in CSV file format (without the header row). A row stands for the parameters of a shot: [body-arm angle],[elbow angle],[shot result].

The shot result is a digit: 0 (miss), 1 (make) or 2 (swish).

Two sample datasets have been included in the extras folder.

Example of a row:

114.84,126.44,1


## Using the program and interpreting the results ##
The use of the program is intuitive and easy: select the text file with the dataset and wait for the results.

Once the calculations have been finished, the optimal parameters and the shot percentage at those parameters will be shown.

In addition, three plots will appear. The first one just plots the selected dataset. The second one shows the most effective range of angles. The third one is a colormap of probabilities, showing the probability of making the shot at different parameters.

## Final notes ##
My intent is to keep working on the application and expand to more than the two parameters mentioned. 

Support is greatly appreciated!
