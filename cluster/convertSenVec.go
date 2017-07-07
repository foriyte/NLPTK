package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	wordVector   map[string][]float64
	taskCount    int
	wordsfile    *os.File
	hashfile     *os.File
	outputfile   *os.File
	idfile       *os.File
	fileLock     sync.Mutex
	allcount     int
	vectorLength int
)

type title struct {
	Id      string
	Content string
}

func main() {
	fmt.Println("start!")
	vectorLength = 50
	taskCount = 100
	allcount = 0
	wordsfile, err := os.Open("data/words.txt")
	if err != nil {
		fmt.Println("open words file error!")
	}
	hashfile, err := os.Open("data/hash.txt")
	if err != nil {
		fmt.Println("open hashid file error!")
	}
	scanner := bufio.NewScanner(wordsfile)
	defer wordsfile.Close()
	scannerid := bufio.NewScanner(hashfile)
	defer hashfile.Close()

	wordVector = loadWordVecDic("data/wordsvec.txt")
	outputfile, err = os.Create("data/senvec.txt")
	if err != nil {
		fmt.Println("create file error!")
	}
	defer outputfile.Close()
	idfile, err = os.Create("data/hash_w2v.txt")
	if err != nil {
		fmt.Println("create hashid file error!")
	}
	defer idfile.Close()
	ch := make(chan title, taskCount)
	for i := 0; i < taskCount; i++ {
		go computeVec(ch)
	}
	count := 0
	for scanner.Scan() && scannerid.Scan() {
		id := scannerid.Text()
		id = strings.Replace(id, "\n", "", -1)
		content := scanner.Text()
		row := title{
			Id:      id,
			Content: content,
		}
		ch <- row
		count += 1
	}
	for allcount < count {
		time.Sleep(2 * time.Second)
	}
}

func computeVec(ch chan title) {
	for {
		row := <-ch
		vec := []float64{}
		for i := 0; i < vectorLength; i++ {
			vec = append(vec, 0.0)
		}
		content := strings.Replace(row.Content, "\n", "", -1)
		for _, token := range strings.Split(content, " ") {
			if token == " " {
				continue
			}
			if _, ok := wordVector[token]; ok {
				addVec(&vec, wordVector[token])
			}
		}
		sum := float64(0.0)
		for i := 0; i < len(vec); i++ {
			sum += vec[i] * vec[i]
		}
		if sum != 0.0 {
			outputline := fmt.Sprintf("")
			for i := 0; i < vectorLength; i++ {
				outputline += fmt.Sprintf("%f ", vec[i])
			}
			outputline += fmt.Sprintf("\n")
			hashline := fmt.Sprintf("%s\n", row.Id)
			fileLock.Lock()
			outputfile.WriteString(outputline)
			idfile.WriteString(hashline)
			allcount += 1
			if allcount%10 == 0 {
				fmt.Printf("have done %d sentences\n", allcount)
			}
			fileLock.Unlock()
		} else {
			fileLock.Lock()
			allcount += 1
			fileLock.Unlock()
		}

	}
}

func loadWordVecDic(filename string) map[string][]float64 {
	inputvec, err := os.Open(filename)
	if err != nil {
		fmt.Println("open vecfile error!")
	}
	defer inputvec.Close()
	res := make(map[string][]float64)
	scanner := bufio.NewScanner(inputvec)

	flag := true
	for scanner.Scan() {
		if flag {
			flag = false
			continue
		}
		content := strings.Replace(scanner.Text(), "\n", "", -1)
		fields := strings.Split(content, " ")
		if len(fields) != vectorLength+1 {
			fmt.Println("length is not match!")
		}
		vec := []float64{}
		for i := 1; i < vectorLength+1; i++ {
			value, err := strconv.ParseFloat(fields[i], 64)
			if err != nil {
				fmt.Println("error")
			}
			vec = append(vec, float64(value))
		}
		res[fields[0]] = vec
	}
	if err := scanner.Err(); err != nil {
		fmt.Println("scanner error")
	}
	return res
}

func addVec(v1 *[]float64, v2 []float64) {
	for i, _ := range *v1 {
		(*v1)[i] = (*v1)[i] + v2[i]
	}
}
