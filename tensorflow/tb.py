import glob
from tensorboard.backend.event_processing import event_accumulator

for name in glob.glob('/home/samsung/mlflow/examples/tensorflow/t2/mlruns/0/*/artifacts/tensorboard_logs/train/events.out.tfevents.*.v2'):
    print (name)
    #ea = event_accumulator.EventAccumulator('/home/samsung/mlflow/examples/tensorflow/t2/mlruns/0/6796880430d844b6b3c04b9f27efbef0/artifacts/tensorboard_logs/train/events.out.tfevents.1620714719.ubuntu.63744.269.v2',   
    ea = event_accumulator.EventAccumulator(name,
       size_guidance={ # see below regarding this argument
       event_accumulator.COMPRESSED_HISTOGRAMS: 500,
       event_accumulator.IMAGES: 4,
       event_accumulator.AUDIO: 4,
       event_accumulator.SCALARS: 0,
       event_accumulator.HISTOGRAMS: 1,
    })

    ea.Reload() # loads events from file


    print(ea.Tags())

