

import os
import numpy as np
import pandas as pd
import json

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def tabulate_events(dpath):

    final_out = []
    # for dname in os.listdir(dpath):
    for dname in glob.glob(dpath):
        # print(f"Converting run {dname}",end="")
        # ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        print(dname)
        ea = EventAccumulator(dname).Reload()
        tags = ea.Tags()['scalars']

        out = {}
        print("")
        for tag in tags:
            tag_values=[]
            wall_time=[]
            steps=[]
            print(tag)
            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)
            out[tag]={"tag_values": tag_values,
            "wall_time": wall_time,
            "steps": steps}

            # out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])
            # print(out)

        final_out.append(json.dumps(out))
        # if len(tags)>0:
        #     df= pd.concat(out.values(),keys=out.keys())
        #     # df.to_csv(f'{dname}.csv')
        #     df.to_csv('1.csv')
        #     print("- Done")
        # else:
        #     print('- Not scalers to write')
        #
        # final_out[dname] = df

    return final_out

if __name__ == '__main__':
    # path = '/home/samsung/mlflow/examples/tensorflow/t2/mlruns/0/*/artifacts/tensorboard_logs/train/events.out.tfevents.*.v2'
    # path = '/home/samsung/mlflow/examples/tensorflow/t3/logs/scalars/*/train/*.v2'
    path = '/home/samsung/mlflow/examples/tensorflow/t3/logs/scalars/events.out.tfevents.1620870266.run2895958-horovod-glwjh'
    # path = '/home/samsung/mlflow/examples/tensorflow/t2/mlruns/0/*/artifacts/tensorboard_logs/train/'
    steps = tabulate_events(path)
    print(steps)
    # pd.concat(steps.values(),keys=steps.keys()).to_csv('all_result.csv')
