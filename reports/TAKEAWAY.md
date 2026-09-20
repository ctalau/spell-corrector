# Takeaway

Am facut mai multe experimente de a antrena un LLM mic sa corecteze erori de spelling. Au fost destul de haotice pt ca nu variam doar un singur parametru la un experiment. 

Am tras niste concluzii:
- Antrenarea unui LLM de la zero are nevoie de multe date pt ca trebuie sa invete si relatiile intre cuvinte nu doar taskul respectiv. 
- LLMs (chiar si ASTRA) nu pare sa aiba intuitii bune legate de LLM training. Mi-a recomandat sa maresc nr de params cand deja facea overfitting (memora exemplele de training). 
- Grok Bot e ca un AI research intern: ii dai task-uri high level si executa experimente.
- Bitter Lesson: LLM merg mai bine fara sugestiile de la Hunspell. (https://en.wikipedia.org/wiki/Bitter_lesson)
- Un model de 0.8B Q4 are un scor de 87% dupa post-training. 
- Modelul de mai sus poate fi rulat pe un CPU cu perf decente.
- Antrenarea unui model mare + distilare da rezultate mai bune ca antrenarea directa. Modelul elev invata si distributia de probabilitate pt alte variante pe langa cea corecta. 
- Acelasi model rulat cu alt stack de servire are alta performanta. 
- Un model de decizie (Jev-style) de 0.6B alege varianta corecta in 94% din cazuri.
- Fine tunning la modele mici e ieftin - 10$ tot experimentul pe RunPod cu community pods.

Oferte de servire
- RunPod pod permanent: 0.3$ pe ora 3090 community
- RunPod serverless: 0.6$. Fara network volume dureaza minute intregi cold start.
- Fireworks serverless: 0.1$ /1M tokens < 4B params (no fine tunning). 
- Fireworks supports models fine tuned on their platform.
