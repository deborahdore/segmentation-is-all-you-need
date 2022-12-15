# segmentation-is-all-you-need
Is segmentation a fundamental step in computer vision? this repo aims to prove that there are problems in which segmentation shouldn't be performed to obtain better results


_Folder structure_:

``` bash
.
├── dataset
│       ├── alfa_romeo
│       ├── bwt
│       ├── ferrari
│       ├── haas
│       ├── mclaren
│       ├── mercedes
│       ├── redbull
│       ├── renault
│       ├── toro_rosso
│       └── williams
└── src
    ├── classification
    │       ├── base
    │       │     ├── config
    │       │     ├── loader
    │       │     ├── model
    │       │     ├── saved
    │       │     │   ├── models
    │       │     │   └── plot
    │       │     └── utils
    │       └── hypertuned
    │                ├── config
    │                ├── loader
    │                ├── model
    │                ├── saved
    │                │   ├── logs
    │                │   ├── models
    │                │   └── plot
    │                └── utils
    ├── detection
    │   ├── config
    │   │   └── xml
    │   ├── saved
    │   │   └── model
    │   ├── utils
    │   └── videos
    └── segmentation
        ├── config
        ├── saved
        │   ├── plot
        │   └── report
        └── utils
```

