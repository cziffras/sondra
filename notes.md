# Notes

So that anyone can quickly grasp the aim of this project I decided to draft this short note.

## The context : why SAR data is peculiar

The context first, SAR imaging is notoriously difficult due to the intrinsic nature of the data : what a SAR sensor receives are the echoes of a signal it previously emitted, they are represented as complex numbers that can fully characterize the phase of a signal. In fact, imagine a signal viewed only through its real part, a cosine, this cosine is periodic so adding a phase to it might not change the appearance of the signal itself.

In practice a SAR receptor keeps on board a stable oscillator that enables performing this simple operation :

$$\text{it emits } e^{j(2\pi f_0 t + \phi)} \quad \longrightarrow \quad \text{it receives } e^{j(2\pi f_0 t + \nu)} \quad \longrightarrow \quad \text{it compares } e^{j(\phi - \nu)}$$

A delay is always due to a traveled distance, since the wave moves at the speed of light this distance $l$ takes $\tau = \frac{l}{c}$ to travel, subsequently, considering the fact that each wavelength traveled is equivalent to one rotation in the complex plane, if the antenna emits $e^{j2\pi f_0 t}$ and receives $e^{j(2\pi f_0 t + \phi)}$ we have :

$$\phi = -2\pi f \tau = -2\pi \frac{l}{\lambda} \quad \text{ with } \quad \lambda = \frac{c}{f}$$

For each target the travel is a round-trip, so $l = 2R$ yielding this final formula :

$$\phi = -\frac{4\pi R}{\lambda}$$

## Recovering the phase without ambiguity : the I and Q components

Let's get back to this quantity $e^{j(\phi - \nu)}$, it is impossible to estimate the value of the difference of $\phi$ and $\nu$ if the sensor was only set to measure the cosine since $\cos(x) = \cos(-x)$, to get rid of this ambiguity, the sensor measures the two components of the signal. How does it perform this computation ? Thanks to a simple mathematical trick, it simply evaluates the real signal in two ways :

Let's suppose the sensor emitted $A \cos(2\pi f_0 t)$ at a frequency $f_0$ with no delay, then receiving the signal, the sensor only perceives $s_r(t) = A \cos(2\pi f_0 t + \phi)$. However we know from the properties of the exponential that :

$$s_r(t) = A \cos(2\pi f_0 t + \phi) = A \cos \phi \, \cos (2\pi f_0 t) - A \sin \phi \, \sin (2\pi f_0 t)$$

Where the two terms of interest $\cos \phi$ and $\sin \phi$ do appear already ! We need to decontaminate them from the varying terms that are still multiplying them, this can be done by considering the fact that for every $T$ a positive real number and $\omega_0 = 2\pi f_0$ the signal's pulsation :

$$\langle \cos^2 (\omega_0 t) \rangle = \frac{1}{T} \int_0^{T} \frac{1 + \cos (2 \omega_0 s)}{2} \, ds = \frac{1}{2}$$

Thanks to $\cos^2(\omega_0 t) = \frac{1 + \cos(2\omega_0 t)}{2}$.

And,

$$\langle \cos (\omega_0 t) \sin (\omega_0 t) \rangle = \frac{1}{T} \int_0^{T} \frac{\sin (2 \omega_0 s)}{2} \, ds = 0$$

When $T$ is such that $\omega_0 T = 2\pi$ or far greater than the period of the signal denoted by $\frac{2\pi}{\omega_0} = \frac{1}{f_0}$, this can be seen for instance in the first case :

$$\langle \cos^2 (\omega_0 t) \rangle = \frac{1}{2} + \frac{1}{4 \omega_0 T} \Big[ \sin(2\omega_0 s) \Big]_0^{T}$$

The second term disappears for a great $T$. However choosing this $T$ might be difficult since the observer is moving and $\phi$ may thus be varying, for that reason, $T$ must be chosen to be greater than the period of the carrier wave but shorter than the characteristic time of variation for $\phi$.

Finally multiplying $s_r$ by the reference $\cos(\omega_0 t)$ and averaging leaves $\tfrac{A}{2}\cos\phi = I$ ; multiplying by $-\sin(\omega_0 t)$ and averaging leaves $\tfrac{A}{2}\sin\phi = Q$. Together they form the complex sample

$$I + jQ = \frac{A}{2} \, e^{j\phi}$$

and $\phi = \operatorname{atan2}(Q, I)$ recovers the angle without ambiguity, since both signs are now known.

## Why a complex-valued architecture

The complex nature of the data is suggesting that we should be implementing a dedicated architecture, possibly with complex parameters that can directly modulate the phase of the input signal. This is all what [`torchcvnn`](https://torchcvnn.github.io/torchcvnn/) is about : this library leverages the [Wirtinger derivative](https://en.wikipedia.org/wiki/Wirtinger_derivatives), which extends differentiation to functions that are *not* complex analytic, which is exactly our case since a real-valued loss over complex weights can never be holomorphic. Writing $z = x + jy$, the two operators are given by :

$$\frac{\partial f}{\partial z} = \frac{1}{2}\left( \frac{\partial f}{\partial x} - j \, \frac{\partial f}{\partial y} \right) \qquad \text{and} \qquad \frac{\partial f}{\partial \bar{z}} = \frac{1}{2}\left( \frac{\partial f}{\partial x} + j \, \frac{\partial f}{\partial y} \right)$$

For a real-valued loss $L$ the direction of steepest descent is carried by the conjugate operator, so the update that a complex optimizer actually applies is :

$$z \longleftarrow z - \eta \cdot 2 \frac{\partial L}{\partial \bar{z}} \quad \text{ with } \quad \frac{\partial L}{\partial \bar{z}} = \overline{\left( \frac{\partial L}{\partial z} \right)}$$

This is the convention [PyTorch follows for complex autograd](https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers), and what `torchcvnn` builds its layers on top of.

This project aimed to use those blocks to construct an efficient contrastive architecture for SAR imaging.

## Why contrastive ?

As we've seen before the nature of the data is peculiar, so visualizing it directly is no treat, from there it appears obvious that annotated data are scarce and certainly not sufficient to train a neural network to perform complex tasks such as semantic segmentation on this type of data.

## The crux : a hierarchical contrastive loss

Now we get to the crux of this project. Most vision architectures that performed well in the literature on semantic segmentation tasks were architectures using different levels of resolution, this is the case for [UNet](https://arxiv.org/abs/1505.04597) or for the [SegFormer](https://arxiv.org/abs/2105.15203) that proved itself having the best results on our data. The problem with those architectures is that they do not in fact compress the input data in a single embedding that could be used with traditional contrastive losses. Plus the fact that the data being complex, we needed to assess whether the hermitian product was itself suited for similarity measurements between vectors. The final issue was to design proper contrastive transformations (transformations that alter an image in positive/negative pairs to be presented to the neural network), they live in [`transforms.py`](src/torchtmpl/transforms/transforms.py). We thus decided to design and test a hierarchical loss performing some sort of weighted sum of classical contrastive losses between the architecture stages. If you want to get more intuition on the SegFormer architecture, a tutorial is available in the source : [`building_segformer.ipynb`](src/torchtmpl/tutorials/building_segformer.ipynb).

## The base loss : NT-Xent on complex embeddings

Before weighing anything we need the per stage loss itself. Each stage produces a batch of $N$ embeddings and their $N$ augmented counterparts, giving $2N$ vectors, and we use the classical NT-Xent on them. Here is where the question raised above gets its answer : the similarity we plug in is the modulus of the hermitian product of the normalized embeddings,

$$\mathrm{sim}(z_i, z_j) = \left| \frac{z_i^{H} z_j}{\lVert z_i \rVert \, \lVert z_j \rVert} \right| \in [0, 1]$$

The hermitian product of two complex vectors is itself complex, so it cannot be fed to a softmax as is, and its argument carries the relative phase between the two embeddings. By keeping only the modulus we get a genuine similarity that is invariant to a global phase rotation, which is exactly what we want : two views of the same target should be considered identical even if the oscillator introduced an arbitrary phase offset between them. With this similarity the loss of a single stage reads

$$L = \frac{1}{2N} \sum_{i=1}^{2N} \left[ -\frac{\mathrm{sim}(z_i, z_{i^{+}})}{\tau} + \log \sum_{j \neq i} \exp \left( \frac{\mathrm{sim}(z_i, z_j)}{\tau} \right) \right]$$

where $i^{+}$ is the index of the positive of $i$ and $\tau$ the temperature. This is implemented in [`NTXentLoss`](src/torchtmpl/losses.py#L85), and it is the brick shared by every variant that follows : what changes from one to the next is only the way the $N_s$ stage losses are combined into a single number.

## First try : the KL loss

The problem with this type of hierarchical loss is that some stages might be noisier than others and their weighing is difficult to assess, tuning it by hand is completely off given the high probability to false results by diminishing the importance of the loss computed for the hardest (and thus the most interesting) stages. To tackle this, the first loss that we designed was leveraging the KL divergence as a simple heuristic for weighing calibration.

We have $N_s$ stages, so $N_s$ losses $L_1, \dots, L_{N_s}$, and we want the network to learn how to combine them rather than fixing the coefficients ourselves. The trick is to hold one free logit $\theta_s$ per stage and to read the weights through a softmax :

$$w = \mathrm{softmax}(\theta) \quad \text{ so that } \quad w_s \geq 0 \quad \text{ and } \quad \sum_{s=1}^{N_s} w_s = 1$$


However, minimizing $\sum_s w_s L_s$ alone would put all the mass on the smallest $L_s$, and a contrastive loss does not only get small when the stage has learned something useful, it also gets small when the representation of that stage collapses. Left alone, our weighing scheme would therefore reward the stage that collapses the fastest, which is the exact opposite of what we are after. This is where the KL divergence comes in, we penalize the departure of $w$ from the uniform distribution $\mathcal{U}$ over the $N_s$ stages :

$$\mathrm{KL}(w \, \Vert \, \mathcal{U}) = \sum_{s=1}^{N_s} w_s \log \frac{w_s}{1/N_s} = \sum_{s=1}^{N_s} w_s \log \left( N_s \, w_s \right)$$

which is non negative, and zero if and only if $w$ is uniform. The final objective is thus

$$\mathcal{L} = \sum_{s=1}^{N_s} w_s L_s + \lambda \, \mathrm{KL}(w \, \Vert \, \mathcal{U})$$

and $\lambda$ becomes the single knob of the whole scheme. Setting it to zero gives back the degenerate case where one stage eats everything, letting it grow brings us back to the plain average of the stages, and in between it decides how much imbalance we are willing to tolerate. Instead of tuning $N_s$ coefficients by hand we are left with one hyperparameter whose meaning is clear.

## Second try : the Kendall loss

The KL scheme does its job, but it remains a heuristic. We picked the uniform distribution as the reference and $\lambda$ as the arbiter, and neither choice is justified by anything deeper than our own judgement. So the natural next question was : instead of designing those weights, could we derive them ? [Kendall and Gal (2018)](https://arxiv.org/abs/1705.07115) answer yes in the multi task setting, and their idea is elegant : if each task carries its own observation noise, the weights stop being free parameters and fall out of a likelihood.

Here is the setup transposed to our case. Our $N_s$ stages share the same encoder of weights $W$, and we assume that each stage $s$ has its own noise level $\sigma_s$, constant over the dataset. This is the homoscedastic hypothesis, the noise depends on the stage and not on the input. With a gaussian likelihood

$$p(y_s \mid f^{W}(x)) = \mathcal{N}\left( f^{W}(x), \, \sigma_s^{2} \right)$$

the log-likelihood of a single observation is

$$\log p(y_s \mid f^{W}(x)) = -\frac{1}{2\sigma_s^{2}} \left\Vert y_s - f^{W}(x) \right\Vert^{2} - \log \sigma_s + \mathrm{cst}$$

and writing $L_s(W) = \Vert y_s - f^{W}(x) \Vert^{2}$, minimizing the joint negative log-likelihood gives

$$\mathcal{L}(W, \sigma) = \sum_{s=1}^{N_s} \frac{1}{2\sigma_s^{2}} L_s(W) + \sum_{s=1}^{N_s} \log \sigma_s$$

The factor $\frac{1}{2\sigma_s^{2}}$ is a sort of precision, so a stage judged noisy sees its own weight go down naturally. The term $\log \sigma_s$ is a barrier, avoiding collapse : sending every $\sigma_s$ to infinity would cancel every term of the sum. 

In practice one parametrizes by $\rho_s = \log \sigma_s^{2}$, which keeps the variances positive by construction, and this is exactly what [`NTXentKendall`](src/torchtmpl/losses.py#L6) holds in its `log_vars` parameter :

$$\mathcal{L} = \sum_{s=1}^{N_s} e^{-\rho_s} L_s + \sum_{s=1}^{N_s} \rho_s$$

The equilibrium is explicit, which is what makes the scheme so appealing on paper. Differentiating the contribution of one stage with respect to $\rho_s$ :

$$\frac{\partial}{\partial \rho_s} \left( e^{-\rho_s} L_s + \rho_s \right) = - e^{-\rho_s} L_s + 1 = 0 \quad \Longrightarrow \quad e^{-\rho_s} = \frac{1}{L_s}$$

so the optimal precision of a stage is the inverse of its own loss. 

## **EDIT** 

As I was writing those notes I realized something wrong in the previous derivation, I detail precisely what below.

### Get back to the second try

Unfortunately at the time I used to see only the similarity part of the NTXent loss that is indeed continuous but not the full picture. The NTXent computes similarities then uses a log likelihood, this is precisely what allowed it to not collapse onto only positive examples : a loss that would only use positive examples would collapse by only predicting the same embedding, this is known from Yann Lecun's research, but using such would be compatible with our previous approach, whence the confusion. 

Looking again at what it is : a log-likelihood over the $2N - 1$ candidates of the batch, that is to say a categorical negative log-likelihood, an InfoNCE. And Kendall's paper does treat that case separately.

$$p(y = c \mid f^{W}(x), \sigma) = \mathrm{Softmax}\left( \frac{1}{\sigma^{2}} f^{W}(x) \right)_c$$

where $\frac{1}{\sigma^{2}}$ sits inside the softmax, as a temperature parameter on the logits. The familiar shape $\frac{1}{\sigma^{2}} \mathrm{CE} + \log \sigma$ only reappears afterwards, through an approximation of the log partition term. 

This is a complete misinterpretation since in the regression case Kendall states that $\frac{1}{2\sigma^2}\Vert y - f\Vert^2 + \log\sigma$ is the negative log-likelihood of an actual model, exactly $-\log p$ for the gaussian $\mathcal{N}(f, \sigma^2)$. In my old code, $L_s = \text{NT-Xent}(\tau_{\text{fixed}})$ is exactly $-\log q_s$ for the categorical law $q_s$ at fixed $\tau$. Let's multiply it by $\beta_s = e^{-\rho_s}$ our weights. The question we should have an answer to (but we did not) is :

Is there a probability law $p$ such that $-\log p = \beta_s \cdot (-\log q_s)$ ?

This would give $p = q_s^{\beta_s}$. But $q_s$ is already a categorical law, so it already sums to one :

- if $\beta > 1$, then $q_k^{\beta} \leq q_k$ for all $k$, we have $\sum_k q_k^{\beta} \leq 1$
- if $\beta < 1$, then $q_k^{\beta} \geq q_k$, so $\sum_k q_k^{\beta} \geq 1$

The equality is only the degenerate case where $\beta = 1$...

So $q_s^{\beta_s}$ is not a law anymore, but renormalizing it fixes the problem. From now on I write $L(\beta)$ for the NT-Xent taken at inverse temperature $\beta$, my old code being frozen at $\beta_0 = 1 / \tau_{\text{fixed}}$ and computing $\beta_s L(\beta_0)$. Renormalizing gives

$$\tilde{p}_j = \frac{q_j^{\beta_s}}{\sum_k q_k^{\beta_s}} \quad \Longrightarrow \quad -\log \tilde{p}_{i^{+}} = \beta_s \left( -\log q_{i^{+}} \right) + \log \sum_k q_k^{\beta_s}$$

and since $q_j \propto e^{\beta_0 \, \mathrm{sim}_j}$ gives $q_j^{\beta_s} \propto e^{\beta_s \beta_0 \, \mathrm{sim}_j}$, this renormalized law is exactly the NT-Xent at inverse temperature $\beta_s \beta_0$ :

$$\underbrace{L(\beta_s \beta_0)}_{\text{a true negative log-likelihood}} = \underbrace{\beta_s \, L(\beta_0)}_{\text{what my old code computed}} + \underbrace{\log \sum_k q_k^{\beta_s}}_{\text{the term I was missing}}$$

So weighing a stage by $\beta_s$ is exactly the same thing as multiplying its inverse temperature by $\beta_s$.

That is perhaps the reason why this loss was not working in practice (we tried fixing it using another regularization term, but the entire principle behind was all flawed), since I hate to stop on a failure, here is, one year later, the third try but this time weighing stages with temperatures : 

## Third try : the learnable temperature

So we go back to the loss we should have used from the start, the categorical one. Each stage gets its own inverse temperature $\beta_s = 1 / \tau_s$, learned like any other parameter, and applied where it belongs, inside the softmax :

$$L_s(\beta_s) = \frac{1}{2N} \sum_{i=1}^{2N} \left[ - \beta_s \, \mathrm{sim}(z_i, z_{i^{+}}) + \log \sum_{j \neq i} \exp \left( \beta_s \, \mathrm{sim}(z_i, z_j) \right) \right]$$

$$\mathcal{L} = \sum_{s=1}^{N_s} L_s(\beta_s) + \lambda \, \mathrm{KL}\left( \tilde{\beta} \, \Vert \, \mathcal{U} \right) \quad \text{ with } \quad \tilde{\beta}_s = \frac{\beta_s}{\sum_k \beta_k}$$

This way, every $L_s(\beta_s)$ remains a genuine negative log-likelihood, the one of the InfoNCE model at temperature $\tau_s$, so their sum is the joint negative log-likelihood of one independent model per stage. There is still a strong assumption of independence but at least the loss itself is well defined !

Why a regularization again ? Because the log partition bounds $\beta_s$ only while the stage is still learning : once it has separated its positives the loss goes flat in $\beta_s$, and if it collapsed the gradient on $\beta_s$ is exactly zero. And what I read is never $\beta_s$ itself, it is how the stages compare, so here comes back the KL against the uniform, this time on the right object !


## **Conclusion**

That is all for this set of notes explaining what was done in this project. I'm glad I could outline this issue in the NTXentKendall loss. Will be adding the results when I have access to a better machine for running one last experiment !
