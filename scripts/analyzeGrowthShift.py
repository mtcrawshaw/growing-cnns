import os
import json
import argparse

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main(experimentName):

    # Read logs
    logPath = os.path.join(projectRoot, 'experiments', experimentName, 
            "%s.log" % experimentName)
    with open(logPath, 'r') as f:
        results = json.load(f)

    growthSteps = max([results['trainResults'][i]['growthStep'] for i in
            range(len(results['trainResults']))])

    differences = []

    # Find increases in loss between growth steps
    for i in range(1, len(results['trainResults'])):
        currentGrowthStep = results['trainResults'][i]['growthStep']
        prevGrowthStep = results['trainResults'][i - 1]['growthStep']

        if currentGrowthStep != prevGrowthStep:
            currentLoss = results['trainResults'][i]['loss']
            prevLoss = results['trainResults'][i - 1]['loss']
            differences.append(currentLoss - prevLoss)

    for i, diff in enumerate(differences):
        print("(%d, %d): %f" % (i, i + 1, diff))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Analyze the increase in the \
            value of the loss function between growth steps.')
    parser.add_argument('experimentName', help='Names of experiment \
            whose log to analyze for loss function increase.')
    args = parser.parse_args()

    main(args.experimentName)
