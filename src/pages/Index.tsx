
import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";

const Index = () => {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="min-h-screen bg-[#0A1428] text-white px-4 py-8">
      <header className="max-w-6xl mx-auto mb-12 text-center">
        <h1 className="text-5xl font-bold mb-4 text-[#E50914]">ReelSense</h1>
        <p className="text-2xl text-gray-300 max-w-3xl mx-auto">
          A complete movie recommendation system using machine learning
        </p>
      </header>

      <div className="max-w-6xl mx-auto">
        <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-3 max-w-xl mx-auto bg-[#192841]">
            <TabsTrigger value="overview" className="text-white">Overview</TabsTrigger>
            <TabsTrigger value="features" className="text-white">Features</TabsTrigger>
            <TabsTrigger value="getStarted" className="text-white">Get Started</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">What is ReelSense?</CardTitle>
                </CardHeader>
                <CardContent>
                  <p>
                    ReelSense is a sophisticated movie recommendation system that uses both content-based 
                    and collaborative filtering approaches to suggest movies that match your preferences.
                  </p>
                  <p className="mt-4">
                    Built with Python's machine learning libraries and integrated with a sleek React interface, 
                    ReelSense helps you discover new films based on what you already love.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">How It Works</CardTitle>
                </CardHeader>
                <CardContent>
                  <p><strong>Content-based filtering:</strong> Recommends movies similar to ones you like based on features like genre, plot, and actors.</p>
                  <p className="mt-2"><strong>Collaborative filtering:</strong> Suggests movies that users with similar tastes have enjoyed.</p>
                  <p className="mt-2"><strong>Hybrid approach:</strong> Combines both methods for more accurate recommendations.</p>
                </CardContent>
              </Card>
            </div>

            <div className="mt-8">
              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">Data-Driven Insights</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-xl font-semibold mb-2">The MovieLens Dataset</h3>
                    <p>
                      This system uses the MovieLens dataset, which contains millions of movie ratings from real users.
                      It's a great resource for training recommendation algorithms and understanding user preferences.
                    </p>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-2">Advanced Analytics</h3>
                    <p>
                      ReelSense performs in-depth exploratory data analysis to uncover patterns in user behavior and movie ratings,
                      helping you understand what makes certain movies popular.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="features" className="mt-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">Content-Based Filtering</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc pl-5 space-y-2">
                    <li>TF-IDF processing of movie descriptions</li>
                    <li>Genre-based similarity analysis</li>
                    <li>Cosine similarity for measuring movie similarity</li>
                    <li>Visualization of movie content relationships</li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">Collaborative Filtering</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc pl-5 space-y-2">
                    <li>Matrix factorization using SVD</li>
                    <li>User-based and item-based approaches</li>
                    <li>Similarity metrics between users and items</li>
                    <li>Personalized rating predictions</li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">Evaluation & Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc pl-5 space-y-2">
                    <li>RMSE (Root Mean Square Error)</li>
                    <li>Precision@K and Recall@K</li>
                    <li>Mean Average Precision (MAP)</li>
                    <li>Normalized Discounted Cumulative Gain</li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">Data Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc pl-5 space-y-2">
                    <li>In-depth exploratory data analysis</li>
                    <li>Rating distribution visualization</li>
                    <li>Genre popularity trends</li>
                    <li>User activity patterns</li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">Interactive UI</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc pl-5 space-y-2">
                    <li>Streamlit web application</li>
                    <li>Movie and user-based recommendations</li>
                    <li>Interactive data visualizations</li>
                    <li>Customizable recommendation parameters</li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="bg-[#192841] border-none text-white">
                <CardHeader>
                  <CardTitle className="text-[#E50914]">Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc pl-5 space-y-2">
                    <li>Efficient data processing</li>
                    <li>Model caching and persistence</li>
                    <li>Scalable architecture</li>
                    <li>Handles large datasets</li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="getStarted" className="mt-8">
            <Card className="bg-[#192841] border-none text-white">
              <CardHeader>
                <CardTitle className="text-[#E50914]">Getting Started with ReelSense</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h3 className="text-xl font-semibold mb-2">1. Clone the Repository</h3>
                    <div className="bg-[#0A1428] p-4 rounded-md">
                      <code>git clone https://github.com/yourusername/reel-sense-recommender.git</code>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xl font-semibold mb-2">2. Install Dependencies</h3>
                    <div className="bg-[#0A1428] p-4 rounded-md">
                      <code>
                        pip install pandas numpy scikit-learn matplotlib seaborn streamlit
                      </code>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xl font-semibold mb-2">3. Download the Dataset</h3>
                    <div className="bg-[#0A1428] p-4 rounded-md">
                      <code>python src/python/download_dataset.py</code>
                    </div>
                    <p className="mt-2 text-sm text-gray-400">This will download the MovieLens dataset.</p>
                  </div>

                  <div>
                    <h3 className="text-xl font-semibold mb-2">4. Launch the Streamlit App</h3>
                    <div className="bg-[#0A1428] p-4 rounded-md">
                      <code>streamlit run src/python/app.py</code>
                    </div>
                  </div>

                  <div className="pt-4">
                    <Button 
                      className="bg-[#E50914] hover:bg-[#B2070E] text-white w-full md:w-auto"
                      onClick={() => window.open("https://github.com/yourusername/reel-sense-recommender", "_blank")}
                    >
                      View Source Code
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      <footer className="max-w-6xl mx-auto mt-16 text-center text-gray-400">
        <p>
          Built with Python, scikit-learn, pandas, and Streamlit | UI: React, Tailwind CSS
        </p>
      </footer>
    </div>
  );
};

export default Index;
