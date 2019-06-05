{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "\n",
    "class GaussianNoise(Dataset):\n",
    "    \"\"\"Gaussian Noise Dataset\"\"\"\n",
    "\n",
    "    def __init__(self, size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0):\n",
    "        self.size = size\n",
    "        self.n_samples = n_samples\n",
    "        self.mean = mean\n",
    "        self.variance = variance\n",
    "        self.data = np.random.normal(loc=self.mean, scale=self.variance, size=(self.n_samples,) + self.size)\n",
    "        self.data = np.clip(self.data, 0, 1)\n",
    "        self.data = self.data.astype(np.float32)\n",
    "\n",
    "    def __len__(self):\n",
    "        return self.n_samples\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        return self.data[idx]\n",
    "\n",
    "\n",
    "class UniformNoise(Dataset):\n",
    "    \"\"\"Uniform Noise Dataset\"\"\"\n",
    "\n",
    "    def __init__(self, size=(3, 32, 32), n_samples=10000, low=0, high=1):\n",
    "        self.size = size\n",
    "        self.n_samples = n_samples\n",
    "        self.low = low\n",
    "        self.high = high\n",
    "        self.data = np.random.uniform(low=self.low, high=self.high, size=(self.n_samples,) + self.size)\n",
    "        self.data = self.data.astype(np.float32)\n",
    "\n",
    "    def __len__(self):\n",
    "        return self.n_samples\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        return self.data[idx]\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
