# sample0 事件状态误判逐条审计

来源：`results/event_state_sample0_v2/dynamic_memory_event_state.pkl`。
人工对照每条 event 的旧、新 fact 引文及原始 Chunk；这是误判诊断，不是数据集标准答案测评。

## 结论

75 条事件包含 73 条 change、2 条 correction；其中 2 条 change 状态为 revoked，仍保留在审计历史。
按‘新的值是否明确替换同一持续状态的旧值’审查，73 条可明确判为误判或无效纠正，
另 2 条（#7、#53）可能是计划具体化，但证据不足，不能认定为已证实变化。
因此本批样例里没有一条可依据当前引文确认的高层状态变化。

| 分类 | 数量 | 编号 |
| --- | ---: | --- |
| 可并存的活动或行为 | 15 | #1, #2, #8, #9, #10, #14, #15, #22, #25, #27, #29, #34, #46, #47, #69 |
| 多值偏好、价值、身份或支持对象 | 16 | #11, #17, #19, #20, #21, #26, #35, #36, #44, #45, #48, #61, #63, #71, #72, #75 |
| 不同照片、创作或拥有物 | 17 | #4, #5, #18, #23, #31, #40, #41, #42, #50, #51, #52, #58, #59, #60, #62, #70, #74 |
| 谈论、提及或泛关系的对象变化 | 22 | #3, #6, #12, #13, #16, #24, #28, #30, #32, #33, #37, #39, #49, #54, #56, #57, #64, #65, #66, #67, #68, #73 |
| 同义细化或无效纠正 | 3 | #38, #43, #55 |
| 可能演化但缺少同一状态的替换证据 | 2 | #7, #53 |

## 关键原因

1. `EventStateLayer.apply()` 使用 `(entity_id, attribute)` 作为单值当前状态槽；同一槽的新值只要字符串不同就被自动写成 change。`attended`、`owns`、`talked_about` 等本是可多值的关系。
2. LLM 把图关系词沿用为状态属性，并把可并存经历标记为 `state_scope=exclusive`；此标记由同一个模型自报，程序仅做字符串检查。
3. 程序验证了引文存在、说话人和图引用，但没有验证新引文是否否定或取代旧引文；纠正分支只检查旧值字符串出现，没有检查语义上的否定。
4. 宽泛属性与实体归属错误造成额外串槽，例如收养机构帮助 LGBTQ+ 人群被记成 Caroline 的 `supports`，以及把‘加入组织’记入 `attended`。
5. 字符串不等也会把同义词或具体化误写为变化，例如 `Guitar → Acoustic guitar`。纠正后重建链还可能把同一误判再次记录（#26、#45）。

## 全部 75 条记录

| # | 事件 ID | 来源 | 实体 / 属性 | 旧值 → 新值 | 判定 | 补充 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `93fffbd5c8d70d847ca7e434` | D3/C18 | Caroline / `attended` | LGBTQ support group → School event | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 2 | `59e28f0bcc3ce563f818be2a` | D3/C21 | Caroline / `current_activity` | go do some research → give others hope | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 3 | `d5d3ca064f56934ae40b71d9` | D3/C21 | Melanie / `related_to` | Mental health → Caroline | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 4 | `08fc8fb4ac985ecaf796f78a` | D4/C31 | Melanie / `attached_to` | sunset → Bowls | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 5 | `f5c1d1796fded20e57cef94b` | D4/C32 | Caroline / `owns` | Necklace → Hand-painted bowl | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 6 | `02737c1519176d39222a90e6` | D4/C34 | Caroline / `talked_about` | transgender journey → Family moments | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 7 | `b3645ceb9829f13d70483dd2` | D4/C36 | Caroline / `career_interest` | check out career options → working with trans people, helping them accept themselves and supporting their mental health | 可能演化但缺少同一状态的替换证据 | 从探索职业选项变为考虑服务跨性别群体，可能是计划具体化；新话语仍说‘I'm thinking’，未明确作出决定。 |
| 8 | `fc0a3ea4fb6636d56524f348` | D4/C36 | Caroline / `attended` | School event → LGBTQ+ counseling workshop | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 9 | `4e861c0d6a83c42867cb5b00` | D5/C39 | Caroline / `attended` | LGBTQ+ counseling workshop → LGBTQ+ pride parade | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 10 | `f50aaa658e8145cff01bba4a` | D5/C41 | Caroline / `current_activity` | give others hope → learning the piano | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 11 | `71d327d3c348bb8a367e2364` | D5/C41 | Melanie / `interest` | Painting → Pottery | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 12 | `17a1c3419d57b3868f3e5fed` | D5/C44 | Caroline / `talked_about` | Family moments → Art | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 13 | `f6fff2a6338b0114c58ee771` | D5/C44 | Melanie / `talked_about` | Family time → Pottery | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 14 | `1161852ba23e72a5ddadd5fe` | D5/C45 | Caroline / `attended` | LGBTQ+ pride parade → transgender conference | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 15 | `4d4895e04a55a122f4d6ac84` | D6/C48 | Melanie / `attended` | Charity race → Museum | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 16 | `193c4bfeeb8b3d0d5e55829e` | D6/C54 | Melanie / `talked_about` | Pottery → Family | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 17 | `576137718cc8dfb6e1da6af4` | D7/C56（已撤销） | Caroline / `supports` | LGBTQ+ → LGBTQ rights | 多值偏好、价值、身份或支持对象 | 旧引文‘they help LGBTQ+ folks with adoption’说的是机构，不是 Caroline；此条已 revoked。 |
| 18 | `db40ce5e7a3cd3c542f701e2` | D7/C62 | Melanie / `owns` | Bowl → Dog | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 19 | `118bb71f3f1a15fca29a9fd5` | D7/C65 | Melanie / `interested_in` | Animals → Running | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 20 | `82a76938515df184aa3f9746` | D7/C67 | Caroline / `values` | Family → Mental health | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 21 | `88f8527af5d6841c214566b5` | D7/C67 | Melanie / `values` | Family → Mental health | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 22 | `a7e120791ba7ee87d3513a29` | D8/C69 | Melanie / `attended` | Museum → Pottery workshop | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 23 | `74878db5cdf483a6ea967c98` | D8/C72 | Melanie / `created` | Bowl → Painting | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 24 | `8529937f0eb028d0f7bd34fc` | D8/C85 | Melanie / `talked_about` | Family → Forest | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 25 | `e8b1d104fce44885b52f3eb6` | D9/C89 | Caroline / `participated_in` | Picnic → Mentorship program | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 26 | `0047a03bad9eeae3cdaa55d9` | D9/C90（已撤销） | Caroline / `supports` | LGBTQ rights → Young folks | 多值偏好、价值、身份或支持对象 | 支持年轻人不意味着不再支持 LGBTQ 权益；此条已 revoked，后由 #45 重建。 |
| 27 | `df9ffc708da6da1dac076f73` | D9/C92 | Caroline / `attended` | transgender conference → Pride event | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 28 | `a573b100ecd5eb2dbde6dbd8` | D9/C97 | Melanie / `talked_about` | Forest → Caroline | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 29 | `2ab6a7abe4537bb02ca3c79c` | D10/C99 | Caroline / `attended` | Pride event → LGBTQ activist group | 可并存的活动或行为 | 把‘joined a new ... group’放进 attended 槽，混淆加入组织和参加活动。 |
| 30 | `b6b6586a44c79d6ec4c1f740` | D10/C101 | Caroline / `talked_about` | Art → Pride Parade | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 31 | `e202c8e3407865cc2c51437e` | D10/C101 | Melanie / `attached_to` | Bowls → Kite | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 32 | `081fb578a5428a10ee2fc8f6` | D10/C106 | Melanie / `talked_about` | Caroline → Life | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 33 | `c0ec22429b6c8c6bf166b76d` | D11/C110 | Melanie / `experienced` | Meteor shower → Summer breeze | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 34 | `1d277848a16138d7e8ed7a8f` | D11/C111 | Caroline / `attended` | LGBTQ activist group → Pride Parade | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 35 | `8b0693a34722dff9d6acef63` | D11/C111 | Caroline / `feeling_in_support_group` | accepted → proud | 多值偏好、价值、身份或支持对象 | 被接纳与感到自豪可以同时成立，而且来自不同场景。 |
| 36 | `b8da80280ef5d1337206ca78` | D11/C111 | Caroline / `values` | Mental health → equality | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 37 | `e28b42461bce3a399f07dc96` | D11/C111 | Melanie / `mentioned` | Stars → Matt Patterson | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 38 | `2296e0aac02e46e96cca79d0` | D11/C116（纠正） | Caroline / `related_to` | Transition → Changing body | 同义细化或无效纠正 | 新引文仍说‘my transition’，没有否定旧的 transition 事实；却被判为 correction。 |
| 39 | `eb80eabb40448a611424a319` | D12/C123 | Caroline / `talked_about` | Pride Parade → Life | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 40 | `36efe96c14c3738d69c84964` | D13/C130 | Caroline / `attached` | Photo → Photo of a sign with a picture of a guinea pig | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 41 | `c0350b0489ec1704e5fc1676` | D13/C131 | Caroline / `owns` | Hand-painted bowl → Oscar | 不同照片、创作或拥有物 | 拥有手绘碗与养豚鼠可同时成立。 |
| 42 | `c9b7d5f96a57b20113ecc218` | D13/C131 | Melanie / `owns` | Dog → Bailey | 不同照片、创作或拥有物 | 原文明确说‘another cat’，增加宠物并未替换原有的狗。 |
| 43 | `ab89c82350a6cffe3c0e7d3c` | D13/C137（纠正） | Caroline / `supports` | LGBTQ rights → LGBTQ rights | 同义细化或无效纠正 | 旧、新值均为 LGBTQ rights；引文没有否定旧事实，却被判为 correction。 |
| 44 | `f1a6c4c3c3dddc56d4c73b52` | D13/C137 | Caroline / `supports` | LGBTQ+ → LGBTQ rights | 多值偏好、价值、身份或支持对象 | 旧引文描述收养机构帮助 LGBTQ+ 人群，主体归属也不成立。 |
| 45 | `6cd0a420de1f517313d8cf5f` | D9/C90 | Caroline / `supports` | LGBTQ rights → Young folks | 多值偏好、价值、身份或支持对象 | 纠正 #43 后重建的事件，仍把两个可并存的支持对象当成互斥。 |
| 46 | `89b00ea2dc04963f11edc2da` | D14/C139 | Caroline / `participated_in` | Mentorship program → Hiking | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 47 | `45514cda666c39bc5601472e` | D14/C142 | Caroline / `visited` | LGBTQ center → Beach | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 48 | `d6d08f818375e00f88f15051` | D14/C145 | Caroline / `values` | equality → Mural | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 49 | `82e6ff3dae9dd3d4d764b5f0` | D14/C145 | Caroline / `associated_with` | 18th birthday → Eagle | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 50 | `545183d07c3191740dc2b498` | D14/C146 | Melanie / `attached_to` | Kite → Stained Glass Window | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 51 | `02188ead1c51c5b51b08200a` | D14/C151 | Caroline / `attached` | Photo of a sign with a picture of a guinea pig → Photo of a painting of flowers and a watercolor palette | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 52 | `39abd1f8cea9920c89783084` | D14/C152 | Caroline / `created` | Stained Glass Window → Drawing | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 53 | `d6a51ef942198937a2e962c4` | D14/C155 | Caroline / `planned_to_do_something_special_in` | Summer → Next Month | 可能演化但缺少同一状态的替换证据 | ‘夏天计划特别的事’和‘下个月办艺术展’可能有关，但原文没有说明后者替换前者。 |
| 54 | `272a34fb2edeeb041d1830d7` | D15/C164 | Melanie / `mentioned` | Matt Patterson → Summer Sounds | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 55 | `f788decaaa81862ebbe89c2a` | D15/C167 | Caroline / `plays` | Guitar → Acoustic guitar | 同义细化或无效纠正 | Acoustic guitar 是 Guitar 的具体种类；且‘五年前开始’不是后来换乐器的证据。 |
| 56 | `66dc1f0ddd2f95dcb8afd98e` | D15/C168 | Melanie / `related_to` | Caroline → Piano | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 57 | `49a5054d32287cca65dc5bbe` | D15/C168 | Melanie / `talked_about` | Life → Caroline | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 58 | `bf06ab632aaca6cef57ec7a9` | D15/C169 | Melanie / `attached_to` | Stained Glass Window → Sheet music | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 59 | `ba26af2ea49ed2b31dbfd64d` | D16/C176 | Caroline / `attached` | Photo of a painting of flowers and a watercolor palette → Photo of a painting on a easel with a red and blue background | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 60 | `e18df62f247dcb0a171dac92` | D16/C177 | Caroline / `created` | Drawing → Painting | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 61 | `8d418bf36e706640e674b5f1` | D16/C177 | Caroline / `identifies_with` | Single parent → Trans woman | 多值偏好、价值、身份或支持对象 | 单亲身份与跨性别女性身份可以同时成立。 |
| 62 | `ab24571e06d38dd8f75fdff0` | D16/C178 | Melanie / `attached_photo_of` | Playground → Sign | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 63 | `8ec257cff8563719ad632401` | D17/C183 | Caroline / `supports` | Young folks → Family | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 64 | `f250c36aebf59b1bc70a7736` | D17/C185 | Caroline / `mentioned` | Advocacy event → Pottery | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 65 | `7093d90e7c0ddf66cb158ec1` | D17/C185 | Melanie / `read` | Charlotte's Web → Book | 谈论、提及或泛关系的对象变化 | ‘that book’未指明书名，不能据此推断她停止阅读 Charlotte's Web。 |
| 66 | `c1061c07df9f5a557e18fe14` | D17/C186 | Melanie / `inspired_by` | Autumn → Sunset | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 67 | `8acf88b7f751be78a10e3242` | D17/C188 | Melanie / `associated_with` | Sunset → Blue | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 68 | `5f6efd7265dd05d353983572` | D17/C189 | Caroline / `related_to` | Changing body → Art | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 69 | `acaa201da6be7bdc6d80b971` | D17/C190 | Caroline / `attended` | Pride Parade → Transgender Poetry Reading | 可并存的活动或行为 | 新引文未说明旧值结束或被取代。 |
| 70 | `3f90a3f5f5f74cea63c1a465` | D17/C190 | Caroline / `attached_photo_of` | Building → Sign | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 71 | `0e73dd605000f2c3276324a5` | D18/C195 | Caroline / `values` | Mural → Family | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 72 | `1a3ad9c46821b4cff3b86979` | D18/C196 | Melanie / `values` | Mental health → Family | 多值偏好、价值、身份或支持对象 | 新引文未说明旧值结束或被取代。 |
| 73 | `eb90310343bdc8cce8b852c9` | D18/C205 | Melanie / `mentioned` | Summer Sounds → Nature | 谈论、提及或泛关系的对象变化 | 新引文未说明旧值结束或被取代。 |
| 74 | `ce0808f0c0012010e706c0ae` | D19/C211 | Caroline / `attached_photo_of` | Sign → Clock | 不同照片、创作或拥有物 | 新引文未说明旧值结束或被取代。 |
| 75 | `9a89f1e4e190b71cbdec3e3b` | D19/C212 | Caroline / `feels_supported_by` | Community → Support | 多值偏好、价值、身份或支持对象 | Community 与 close people 的支持可并存，Support 又是泛称，不能作替换。 |

逐条完整引文、对应事实及原始 Chunk 另见 `event_review.json`。
