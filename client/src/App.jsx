import { useEffect, useState } from 'react'
import './App.css'
import axios from 'axios';


function App() {

//setup for form for ml model
  const [bmi, setBmi] = useState(null);
  const [height, setheight] = useState(null);
  const [weight, setWeight] = useState(null)
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [familyHistory, setFamilyHistory] = useState('');
  const [diagnosis, setDiagnosis] = useState(null);



//setup of communication for iot machine
  const apiKey = 'A874YWYHQAIBUIBD';
  const channelId = '2548174';
  const url = `https://api.thingspeak.com/channels/${channelId}/feeds/last.json?api_key=${apiKey}`;

  
//funciton to handle form submit
  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = {
        Age: parseFloat(age),
        Gender: gender,
        BMI: parseFloat(bmi),
        'Family History of Diabetes': familyHistory
    };
    try {
      const response = await axios.post('http://localhost:5000/predict', data);
      setDiagnosis(response.data.Diagnosis);
  } catch (error) {
      console.error("There was an error making the request:", error);
  }
};
  

//function to fetch iot data
    const fetchData = async () => {
        try {
            const response = await axios.get(url);
            console.log(`bmi is ${response.data.field3}`)
            console.log(`weight is ${response.data.field2}`)
            console.log(`height is ${response.data.field1}`)
            setBmi(response.data.field3)
            setWeight(response.data.field2)
            setheight(response.data.field1)
            // setData(response.data);
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    };

//function to reset the prediction
const resetpredicton = () => {
  setBmi(null)
  setheight(null)
  setWeight(null)
  setGender('')
  setAge('')
  setDiagnosis('')
  setFamilyHistory('')
  
} 


  

  return (
    <>
      <div className='app'>
      
        <h1>Diabetes Prediction Model</h1>
        <button onClick={fetchData}>get data from IOT machine</button>
        {bmi === null ? (
                <p>Click the above button to get bmi from iot machine </p>
            ) : (
                <>
                  <p>Your BMI is: {bmi}</p>
                  <p>Your height is: {height}</p>
                  <p>Your weight is: {weight}</p>
              </>
              
            )}
           
            <div className="mlmodel">
            <form onSubmit={handleSubmit} className='form'>
                <label className='labels'>
                    Age:
                    <input type="number" value={age} onChange={(e) => setAge(e.target.value)} required />
                </label>
                <label className='labels'>
                    Gender:
                    <select value={gender} onChange={(e) => setGender(e.target.value)} required>
                        <option value="">Select...</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </label>
                
                <label className='labels'>
                    Family History of Diabetes:
                    <select value={familyHistory} onChange={(e) => setFamilyHistory(e.target.value)} required>
                        <option value="">Select...</option>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </label>
                <div>
                <button type="submit">Predict</button>
                <button onClick={resetpredicton}>reset</button>
                </div>
            </form>
            {diagnosis && (
                <div>
                    <h2>Diagnosis: {diagnosis}</h2>
                </div>
            )}
            </div>


      </div>
    </>
  )
}

export default App
