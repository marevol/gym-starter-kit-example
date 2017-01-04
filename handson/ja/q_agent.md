# エージェント開発 (Q学習編)

ここでは、Q学習(Q Learning)を用いて、学習するエージェントを作成します。

## Q学習とは

Q学習とは、モデルフリーな強化学習の方法です。
簡単に説明すると、状態`s`と行動`a`から決まる行動評価`Q(s,a)`を用いて、評価が高い行動を選択していく方法です。
`Q(s,a)`は、状態`s`のときに行動`a`がどの程度適切だったかを表す関数であり、`Q(s,a)`を適切に学習していくことで、`Q(s,a)`が最も高くなる行動`a`を選択することができるようになります。

たとえば、状態`s_t`のエージェントが行動`a`を選択して、状態が`s_{t+1}`に遷移した場合、`Q(s_t,a)`を次式で更新する。

![Q(s\_t,a)](https://wikimedia.org/api/rest_v1/media/math/render/svg/1530febfe82181f7d15ff3bc85c9c04e3ebe8d1c)

これにより、1ステップごとに`Q(s,a)`の値を更新していくことで、エージェントが適切な行動を選択できるようにしていく。

## QAgent

Q学習を実装したエージェントを作成しましょう。
ベースとなるエージェントはagent/q\_agent.pyに用意してあります。
GymKitAgentを継承していますが、RandomAgentと比べて、以下のメソッドがオーバーライドして追加しています。

* `def fit(self, observation, reward, done, info):`
  act後に環境から返却される情報が渡されます。
* `def __enter__(self):`
  エピソードの開始時に呼び出されます。
* `def __exit__(self, exception_type, exception_value, traceback):`
  エピソードの終了時に呼び出されます。

まず、`Q(s,a)`の更新処理をfitメソッド内に記述してください。

```python
        future = 0 if done else np.max(current_actions)
        value = previous_actions[self.action]
        previous_actions[self.action] += self.alpha * (reward + self.gamma * future - value)
```

OpenAI Gymの環境からの観測結果はobservationとして渡されます。
Q学習では観測結果から状態を判断する必要がありますが、今回はobservationを離散化することで状態として扱います。
以下のメソッドでobservationからstateを取得するので、記述してください。

```python
    def observation_to_state(self, observation):
        state = ''
        bins = self.get_state_bins()
        for d, o in enumerate(observation.flatten()):
            state += str(np.digitize(o, bins[d]))
        return state

    def get_state_bins(self, bin_sizes=[2, 2, 7, 4]):
        if hasattr(self, '_dimension_bins'):
            return self._dimension_bins

        self._dimension_bins = []
        lows = self.env.observation_space.low
        highs = self.env.observation_space.high
        for i in range(len(lows)):
            low = lows[i]
            high = highs[i]

            r = (float(high) - float(low)) / (bin_sizes[i] - 1)
            bins = np.arange(low + r / 2, high, r)
            if min(bins) < 0 and 0 not in bins:
                bins = np.sort(np.append(bins, [0]))
            self._dimension_bins.append(bins)

        return self._dimension_bins
```

### 実行

QAgentは以下のコマンドで実行することができます。

    $ gym-start --agent agent.QAgent --try-count 1000

--try-countオプションでエピソードの実行回数を指定します。
ここでは1000回のエピソードを実行するように指定しています。

### 補足

TBD: パラメータ


