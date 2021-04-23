(defun whitespace-char(ch)
	(member ch '(#\Space #\Tab #\Newline))
)

(defun russian-upper-case-p (char)
    (position char "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
)

(defun russian-char-downcase (char)
	(let ((i (russian-upper-case-p char)))
        (if i 
            (char "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" i)
            (char-downcase char)
        )
    )
)  

(defun russian-char-equal (char1 char2)
    (char-equal (russian-char-downcase char1)
        (russian-char-downcase char2))
)

(defun endSentence-char(ch)
	(member ch '(#\! #\? #\. #\"))
)

(defun count-words-with-start-eq-end(str)
	(let
		(
			(res 0)
            (i 0)
            (j 0)
			(cur-ch-begin nil)
			(cur-ch-end nil)
		)
		
		(loop while (and (not (endSentence-char (char str i))) (< i (length str))) do
			(setq cur-ch-begin (char str i))
			(setq j i)
            (loop while (and (< j (length str)) (and (not (whitespace-char (char str j))) (not (endSentence-char (char str j))))) do
                (setq j (+ j 1))
            )
            (setq j (- j 1))
            (setq cur-ch-end (char str j))
			(if (or (char-equal cur-ch-begin cur-ch-end) (russian-char-equal cur-ch-begin cur-ch-end))
				(setq res (+ res 1))
			)
			(setq i (+ j 1))
            (loop while (and (< i (length str)) (and (not (endSentence-char (char str i))) (whitespace-char (char str i)))) do
                (setq i (+ i 1))
            )
		)
        (write res)
	)
)