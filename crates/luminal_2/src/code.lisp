; -------- SYMBOLIC ALGEBRA -------
(ruleset expr)
(datatype Expression
	(MNum i64)
	(MVar String)
	(MAdd Expression Expression)
	(MSub Expression Expression)
	(MMul Expression Expression)
	(MDiv Expression Expression)
	(MMod Expression Expression)
	(MMin Expression Expression)
	(MMax Expression Expression)
	(MAnd Expression Expression)
	(MOr Expression Expression)
	(MGte Expression Expression)
	(MLt Expression Expression)
	(MFloorTo Expression Expression)
    (MReplace Expression Expression Expression)
    (MAccum String) ; this marks that we feed the output (also marked with MAccum) back in
)

; Communative
(rewrite (MAdd a b) (MAdd b a) :ruleset expr)
(rewrite (MMul a b) (MMul b a) :ruleset expr)

; Associative
(rewrite (MAdd (MAdd a b) c) (MAdd a (MAdd b c)) :ruleset expr)
(rewrite (MMul (MMul a b) c) (MMul a (MMul b c)) :ruleset expr)

; Constant folding
(rewrite (MAdd (MNum a) (MNum b)) (MNum (+ a b)) :ruleset expr)
(rewrite (MSub (MNum a) (MNum b)) (MNum (- a b)) :ruleset expr)
(rewrite (MMul (MNum ?a) (MNum ?b)) (MNum (* ?a ?b)) :ruleset expr) ; can this overflow?
(rewrite (MDiv (MNum a) (MNum b)) (MNum (/ a b)) :when ((!= 0 b) (= 0 (% a b))) :ruleset expr)
(rewrite (MMax (MNum a) (MNum b)) (MNum (max a b)) :ruleset expr)
(rewrite (MMin (MNum a) (MNum b)) (MNum (min a b)) :ruleset expr)
(rewrite (MAnd (MNum a) (MNum b)) (MNum (& a b)) :ruleset expr)

; Simple reductions
(rewrite (MAdd a (MNum 0)) a :ruleset expr)
(rewrite (MMul a (MNum 1)) a :ruleset expr)
(rewrite (MMul a (MNum 0)) (MNum 0) :ruleset expr)
(rewrite (MDiv a (MNum 1)) a :ruleset expr)
(rewrite (MMod (MMul ?x ?y) ?y) (MNum 0) :ruleset expr)
(rewrite (MMod (MMod ?x (MNum ?y)) (MNum ?z)) (MMod ?x (MNum ?y)) :when ((>= ?z ?y) (= 0 (% ?y ?z))) :ruleset expr) ; nested mods
(rewrite (MMod (MMod ?x (MNum ?y)) (MNum ?z)) (MMod ?x (MNum ?z)) :when ((>= ?y ?z) (= 0 (% ?z ?y))) :ruleset expr)

; Replacement
(rewrite (MReplace ?x ?y ?z) ?z :when ((= ?x ?y)) :ruleset expr)
(rewrite (MReplace (MAdd ?a ?b) ?x ?y) (MAdd (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MSub ?a ?b) ?x ?y) (MSub (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMul ?a ?b) ?x ?y) (MMul (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MDiv ?a ?b) ?x ?y) (MDiv (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMod ?a ?b) ?x ?y) (MMod (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMin ?a ?b) ?x ?y) (MMin (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMax ?a ?b) ?x ?y) (MMax (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MFloorTo ?a ?b) ?x ?y) (MFloorTo (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
; leave numbers unchanged
(rewrite (MReplace (MNum ?n) ?x ?y) (MNum ?n) :ruleset expr)
(rewrite (MReplace (MAccum ?acc) ?x ?y) (MAccum ?acc) :ruleset expr)

; leave other vars unchanged
(rewrite (MReplace (MVar ?v) (MVar ?x) ?y) (MVar ?v) :when ((!= ?v ?x)) :ruleset expr)

; -------- IR --------
(ruleset ir)
(ruleset ir-prop)
(datatype UnOp
	(Exp2)
	(Log2)
	(Sqrt)
	(Sin)
	(Recip)
	(Neg)
)
(datatype BinOp
	(Add)
	(Mul)
	(Max)
)
(sort ESet (Set Expression))
(datatype IR
   	(GMEM String)
   	(LoopIn IR Expression Expression)
   	(LoopOut IR Expression Expression)
    (Unary UnOp IR)
   	(Binary BinOp IR IR)

   	; propogation patterns
   	(SwapLoops IR i64) ; Swap two loops, identified by the inner loop level
   	(TileLoop IR i64) ; Tile a loop, identified by it's loop level
    (MergeLoops IR i64) ; Merge loops, identified by the inner loop level
    (Fused IR ESet) ; Says that we have previously fused loopout -> loopins here

   	; tensor core stuff
   	(TCMatmul IR IR Expression Expression Expression Expression Expression Expression) ; input A, input B, A k stride, B k stride, A inner stride, B inner stride, C inner stride, number of K tile loops
   	(TiledMatmulInputA IR i64 Expression)
    (TiledMatmulInputB IR i64 Expression)
)

; -------------- HELPERS ---------------
(sort ExpressionSetBase (Set Expression))
(sort IRSetBase (Set IR))

(function MAccumSet () ExpressionSetBase :merge (set-union old new))
(rule
	((= ?e (MAccum ?s)))
	((set (MAccumSet) (set-of ?e)))
	:ruleset ir-prop
)
(function LoopInSet () IRSetBase :merge (set-union old new))
(rule
	((= e (LoopIn x y z)))
	((set (LoopInSet) (set-of e)))
	:ruleset ir-prop
)

(function loop_level (IR) i64 :merge new)
; GMEM (0) -> loopin (0)
(rule
	(
		(= out (LoopIn g l1 r1))
		(= g (GMEM x))
	)
	(
		(set (loop_level out) 0)
		(set (loop_level g) 0)
	)
	:ruleset ir-prop
)
; non-loopin (n) -> loopout (n - 1)
(rule
	(
		(= curr (LoopOut x l1 r1))
		(set-not-contains (LoopInSet) x)
		(= xll (loop_level x))
	)
	((set (loop_level curr) (- xll 1)))
	:ruleset ir-prop
)
; loopin (n) -> loopout (n)
(rule
	(
		(= curr (LoopOut x l1 r1))
		(= x (LoopIn y l2 r2))
		(= xll (loop_level x))
	)
	((set (loop_level curr) xll))
	:ruleset ir-prop
)
; loopin (n) -> loopin (n + 1)
(rule
	(
		(= curr (LoopIn x l2 r2))
		(set-contains (LoopInSet) x)
		(= xll (loop_level x))
	)
	((set (loop_level curr) (+ xll 1)))
	:ruleset ir-prop
)
; loopout (n) -> loopin (n)
(rule
	(
		(= curr (LoopIn x l1 r1))
		(= x (LoopOut y l2 r2))
		(= xll (loop_level x))
	)
	((set (loop_level curr) xll))
	:ruleset ir-prop
)
; non-loopin -> binary
(rule
	(
		(= curr (Binary bin a b))
		(set-not-contains (LoopInSet) a)
		(= xll (loop_level a))
	)
	((set (loop_level curr) xll))
	:ruleset ir-prop
)
; loopin (n) -> binary (n + 1)
(rule
	(
		(= curr (Binary bin a b))
		(= a (LoopIn in1 l1 r1))
		(= xll (loop_level a))
	)
	((set (loop_level curr) (+ xll 1)))
	:ruleset ir-prop
)
; non-loopin -> unary
(rule
	(
		(= curr (Unary un a))
		(set-not-contains (LoopInSet) a)
		(= xll (loop_level a))
	)
	((set (loop_level curr) xll))
	:ruleset ir-prop
)
; loopin (n) -> unary (n + 1)
(rule
	(
		(= curr (Unary un a))
		(= a (LoopIn in1 l1 r1))
		(= xll (loop_level a))
	)
	((set (loop_level curr) (+ xll 1)))
	:ruleset ir-prop
)
; something -> fused
(rule
	(
		(= curr (Fused x y))
		(= xll (loop_level x))
	)
	((set (loop_level curr) (- xll 1)))
	:ruleset ir-prop
)
; loopin -> tcmatmul
(rule
	(
		(= curr (TCMatmul a b c d e f g h))
		(= a (LoopIn x l1 r1))
		(= xll (loop_level a))
	)
	((set (loop_level curr) (+ xll 1)))
	:ruleset ir-prop
)
; loopout -> loopout
(rule
	(
		(= curr (LoopOut x l1 r1))
		(= x (LoopOut y l2 r2))
		(= xll (loop_level x))
	)
	((set (loop_level curr) (- xll 1)))
	:ruleset ir-prop
)

; ---------- RULES ----------

; Loop Fusion
(ruleset fusion)
(rewrite
	(LoopIn (LoopOut ?a ?range ?st) ?range ?st)
	(Fused ?a (set-of ?range))
	:when ((set-not-contains (MAccumSet) ?st))
	:ruleset fusion
)
(rewrite
	(LoopIn (Fused (LoopOut ?a ?range ?st) ?prev_fused) ?range ?st)
	(Fused ?a (set-insert ?prev_fused ?range))
	:ruleset fusion
)

; Tiling
(rule
	(
		(= ?e (LoopOut ?body (MNum ?range) ?stride))
		(= ?ll (loop_level ?e))
		(> ?range 8) ; range must be larger than 8
		(= (% ?range 8) 0) ; range must be divisible by 8
	)
	(
		(union ?e
			(LoopOut
				(LoopOut
					(TileLoop ?body ?ll)
					(MNum 8)
					?stride
				)
				(MNum (/ ?range 8))
				(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 8)))
			)
		)
	)
	:ruleset ir
)
(rule
	(
		(= ?loop (LoopIn ?body (MNum ?range) ?stride))
		(= ?e (TileLoop ?loop ?ll))
		(= ?ll (loop_level ?loop))
	)
	(
		(union ?e
			(LoopIn
				(LoopIn ?body
					(MNum (/ ?range 8))
					(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 8)))
				)
				(MNum 8)
				?stride
			)
		)
	)
	:ruleset ir-prop
)
(rule
	(
		(= ?x (LoopIn ?body ?range ?stride))
		(= ?e (TileLoop ?x ?ll))
		(> (loop_level ?x) ?ll)
	)
	(
		(union ?e (LoopIn (TileLoop ?body ?ll) ?range ?stride))
	)
	:ruleset ir-prop
)
(rewrite
	(TileLoop (LoopOut ?body ?range ?stride) ?ll)
	(LoopOut (TileLoop ?body ?ll) ?range ?stride)
	:ruleset ir-prop
)
(rewrite
	(TileLoop (Unary ?un ?body) ?ll)
	(Unary ?un (TileLoop ?body ?ll))
	:ruleset ir-prop
)
(rewrite
	(TileLoop (Binary ?bin ?bodyA ?bodyB) ?ll)
	(Binary ?bin (TileLoop ?bodyA ?ll) (TileLoop ?bodyB ?ll))
	:ruleset ir-prop
)

; Merging
(rule
	(
		(= ?inner
			(LoopOut ?x
				(MNum ?rangeI) ?stI
			)
		)
		(= ?e
			(LoopOut
				?inner
				(MNum ?rangeO) ?stO
			)
		)
		(= ?ll (loop_level ?inner))
		(set-not-contains (MAccumSet) ?stI)
		(set-not-contains (MAccumSet) ?stO)
	)
	(
		(union ?e
			(LoopOut
				(MergeLoops ?x ?ll)
				(MNum (* ?rangeO ?rangeI))
				(MAdd (MReplace ?stO (MVar "z") (MDiv (MVar "z") (MNum ?rangeI))) (MReplace ?stI (MVar "z") (MMod (MVar "z") (MNum ?rangeI))))
			)
		)
	)
	:ruleset ir
)
(rule
	(
		(= ?inner (LoopIn (LoopIn ?x ?rangeO ?stO) ?rangeI ?stI))
		(= ?ll (loop_level ?inner))
		(= ?e (MergeLoops ?inner ?ll))
	)
	(
		(union
			?e
			(LoopIn
				?x
				(MMul ?rangeO ?rangeI)
				(MAdd (MReplace ?stO (MVar "z") (MDiv (MVar "z") ?rangeI)) (MReplace ?stI (MVar "z") (MMod (MVar "z") ?rangeI)))
			)
		)
	)
	:ruleset ir-prop
)
(rule
	(
		(= ?x (LoopIn ?body ?range ?stride))
		(= ?e (MergeLoops ?x ?ll))
		(> (loop_level ?x) ?ll)
	)
	(
		(union ?e (LoopIn (MergeLoops ?body ?ll) ?range ?stride))
	)
	:ruleset ir-prop
)
(rewrite
	(MergeLoops (LoopOut ?body ?range ?stride) ?ll)
	(LoopOut (MergeLoops ?body ?ll) ?range ?stride)
	:ruleset ir-prop
)
(rewrite
	(MergeLoops (Unary ?un ?body) ?ll)
	(Unary ?un (MergeLoops ?body ?ll))
	:ruleset ir-prop
)
(rewrite
	(MergeLoops (Binary ?bin ?bodyA ?bodyB) ?ll)
	(Binary ?bin (MergeLoops ?bodyA ?ll) (MergeLoops ?bodyB ?ll))
	:ruleset ir-prop
)

; Swapping
(rule
	(
		(= ?inner
			(LoopOut
				?x
				?innerRange
				?innerStride
			)
		)
		(= ?e
			(LoopOut
				?inner
				?outerRange
				?outerStride
			)
		)
		(= ?ll  (loop_level ?inner))
		(set-not-contains (MAccumSet) ?innerStride)
		(!= ?innerRange ?outerRange)
	)
	(
		(union
			?e
			(LoopOut
				(LoopOut
					(SwapLoops ?x ?ll)
					?outerRange
					?outerStride
				)
				?innerRange
				?innerStride
			)
		)
	)
	:ruleset ir
)
(rule
	(
		(= ?inner
			(LoopIn
				(LoopIn
					?x
					?outerRange
					?outerStride
				)
				?innerRange
				?innerStride
			)
		)
		(= ?ll (loop_level ?inner))
		(= ?e (SwapLoops ?inner ?ll))
	)
	(
		(union ?e
			(LoopIn
				(LoopIn
					?x
					?innerRange
					?innerStride
				)
				?outerRange
				?outerStride
			)
		)
	)
	:ruleset ir-prop
)
(rule
	(
		(= ?x (LoopIn ?body ?range ?stride))
		(= ?e (SwapLoops ?x ?ll))
		(> (loop_level ?x) ?ll)
	)
	(
		(union ?e (LoopIn (SwapLoops ?body ?ll) ?range ?stride))
	)
	:ruleset ir-prop
)
(rewrite
	(SwapLoops (LoopOut ?body ?range ?stride) ?ll)
	(LoopOut (SwapLoops ?body ?ll) ?range ?stride)
	:ruleset ir-prop
)
(rewrite
	(SwapLoops (Unary ?un ?body) ?ll)
	(Unary ?un (SwapLoops ?body ?ll))
	:ruleset ir-prop
)
(rewrite
	(SwapLoops (Binary ?bin ?bodyA ?bodyB) ?ll)
	(Binary ?bin (SwapLoops ?bodyA ?ll) (SwapLoops ?bodyB ?ll))
	:ruleset ir-prop
)

; TensorCore
(ruleset tc)
(rewrite
	(LoopIn ; k
		(LoopIn ; n
			(LoopIn ; m
				?a
				(MNum ?m)
				(MMul (MVar "z") (MNum ?k))
			)
			(MNum ?n)
			(MNum 0)
		)
		(MNum ?k)
		(MVar "z")
	)
	(TiledMatmulInputA ?a ?k (MNum (/ ?k 8)))
	:when ((= (loop_level ?a) 0) (= (% ?k 8) 0) (= (% ?m 8) 0) (= (% ?n 8) 0))
	:ruleset tc
)
(rewrite
	(LoopIn ; k
		(LoopIn ; n
			(LoopIn ; m
				?b
				(MNum ?m)
				(MNum 0)
			)
			(MNum ?n)
			(MVar "z")
		)
		(MNum ?k)
		(MMul (MVar "z") (MNum ?n))
	)
	(TiledMatmulInputB ?b ?n (MNum (/ ?k 8)))
	:when ((= (loop_level ?b) 0) (= (% ?k 8) 0) (= (% ?m 8) 0) (= (% ?n 8) 0))
	:ruleset tc
)
(rewrite
	(LoopOut ; m
		(LoopOut ; n
			 (LoopOut ; k
				(Binary (Add)
					(Fused (Binary (Mul)
						(TiledMatmulInputB ?b ?n ?k_loops)
						(TiledMatmulInputA ?a ?k ?k_loops)
					) ?fusedloops)
					; accumulator
					(LoopIn ; k
						(LoopIn ; n
							(LoopIn ; m
								?acc
								(MNum ?m)
								(MNum 0)
							)
							(MNum ?n)
							(MNum 0)
						)
						(MNum ?k)
						(MAccum ?accum)
					)
				)
				(MNum ?k)
				(MAccum ?acc_outer)
			)
			(MNum ?n)
			(MVar "z")
		)
		(MNum ?m)
		(MMul (MVar "z") (MNum ?n))
	)
	(LoopOut ; m outer
		(LoopOut ; n outer
			(LoopOut ; m tile
				(LoopOut ; n tile
					(TCMatmul
						; a
						(LoopIn ; n tile
							(LoopIn ; m tile
								(LoopIn ; n outer
									(LoopIn ; m outer
										?a
										(MNum (/ ?m 8))
										(MMul (MVar "z") (MNum (* ?k 8)))
									)
									(MNum (/ ?n 8))
									(MNum 0)
								)
								(MNum 8)
								(MNum 0)
							)
							(MNum 4)  ; each thread in the matmul does 2 elements
							(MNum 0)
						)
						; b
						(LoopIn ; n tile
							(LoopIn ; m tile
								(LoopIn ; n outer
									(LoopIn ; m outer
										?b
										(MNum (/ ?m 8))
										(MNum 0)
									)
									(MNum (/ ?n 8))
									(MMul (MVar "z") (MNum 8))
								)
								(MNum 8)
								(MNum 0)
							)
							(MNum 4)  ; each thread in the matmul does 2 elements
							(MNum 0)
						)
						; a k stride
						(MMul (MVar "z") (MNum 8))
						; b k stride
						(MMul (MVar "z") (MNum (* ?n 8)))
						; a row size
						(MNum ?k)
						; b row size
						(MNum ?n)
						; c row size
						(MNum ?n)
						; k loops
						?k_loops
					)
					(MNum 4)
					(MNum 0)
				)
				(MNum 8)
				(MNum 0)
			)
			(MNum (/ ?n 8))
			(MMul (MVar "z") (MNum 8))
		)
		(MNum (/ ?m 8))
		(MMul (MVar "z") (MNum (* ?n 8)))
	)
	:ruleset tc
)

(ruleset end)
(rewrite
	(MMul (MNum a) (MDiv b (MNum c)))
	(MMul b (MNum (/ a c)))
	:when ((= (% a c) 0))
	:ruleset end
)


{code}

(run-schedule
	(saturate expr)
	(saturate ir-prop)
	(let-scheduler bo (back-off))
	(repeat 1
		(run-with bo ir)
		(saturate ir-prop)
		(saturate expr)
		(saturate fusion)
	)
	(saturate ir-prop)
	(saturate tc)
	(saturate ir-prop)
	(saturate end)
)

;(print-size)